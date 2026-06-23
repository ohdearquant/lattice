import Darwin
import Foundation

// MARK: - PID registry for active trainer subprocesses.
//
// Survives app crashes: each active run is written to a per-PID JSON file before launch and
// deleted on exit. On next startup, reapOrphans() reads the directory and kills any trainer
// processes that outlived the app session.
//
// Storage layout (single-instance assumed — two live app instances are out of scope):
//   <AppSupport>/LatticeStudio/active-runs/<pid>.json
//
// The "active-runs" subdirectory lives under the SAME base that AppStore already uses for
// runs.json. Both share AppStore.appSupportDir so the path never diverges.

// MARK: Persisted entry

/// Minimal descriptor written for each active subprocess.
struct ActiveRunEntry: Codable {
    let pid: Int32
    let binPath: String
    let kind: String
    let startedAt: Date
}

// MARK: Registry

enum RunRegistry {

    // MARK: Directory resolution

    /// The `active-runs` subdirectory under the shared LatticeStudio application support dir.
    /// Returns nil if the app support dir is inaccessible; all callers handle nil gracefully.
    private static var registryDir: URL? {
        guard let base = AppStore.appSupportDir else { return nil }
        let dir = base.appendingPathComponent("active-runs", isDirectory: true)
        // Create on first access — best-effort, failure propagates as nil.
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            print("[RunRegistry] could not create registry dir: \(error)")
            return nil
        }
        return dir
    }

    private static func entryURL(for pid: Int32) -> URL? {
        registryDir?.appendingPathComponent("\(pid).json")
    }

    // MARK: Registration

    /// Write a PID entry before the subprocess is considered launched.
    /// Failure is best-effort: a missing entry is still caught on next reap.
    static func register(pid: Int32, binPath: String, kind: String, startedAt: Date) {
        guard let url = entryURL(for: pid) else { return }
        let entry = ActiveRunEntry(pid: pid, binPath: binPath, kind: kind, startedAt: startedAt)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(entry) else { return }
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            print("[RunRegistry] failed to register pid \(pid): \(error)")
        }
    }

    /// Remove the PID entry after the subprocess terminates.
    /// Safe to call on a pid that was never registered (or already deregistered).
    static func deregister(pid: Int32) {
        guard let url = entryURL(for: pid) else { return }
        try? FileManager.default.removeItem(at: url)
    }

    // MARK: Startup reaper

    /// Scan the registry dir and terminate any orphaned trainer processes left over from a
    /// previous app session (crash, force-quit, etc.).
    ///
    /// Safety gate: before sending SIGTERM we verify via proc_pidpath that the running
    /// process at that PID is the same executable recorded at registration time.
    /// If the PID was recycled by an unrelated process we remove the stale file and move on.
    ///
    /// Returns the count of processes actually killed (i.e. matching exe-path and successfully
    /// signalled). Processes that are already dead result in stale-file cleanup only.
    ///
    /// Note: assumes single-instance app. A second live instance would incorrectly appear as
    /// orphaned training runs at startup of the first instance; that scenario is out of scope.
    @discardableResult
    static func reapOrphans() -> Int {
        guard let dir = registryDir else { return 0 }
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else {
            return 0
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var killed = 0

        for fileURL in contents {
            guard fileURL.pathExtension == "json" else { continue }
            guard let data = try? Data(contentsOf: fileURL),
                  let entry = try? decoder.decode(ActiveRunEntry.self, from: data) else {
                // Corrupt or undecodable entry — remove to avoid accumulation.
                try? fm.removeItem(at: fileURL)
                continue
            }

            let pid = entry.pid

            // send(0) probes liveness without delivering a signal.
            if kill(pid, 0) != 0 {
                // Process is already gone — clean up the stale registry file.
                try? fm.removeItem(at: fileURL)
                continue
            }

            // Process is alive. Verify the executable path matches before signalling.
            // PROC_PIDPATHINFO_MAXSIZE (4 * MAXPATHLEN) is 4096 on Darwin. The C macro is
            // not importable into Swift ("structure not supported"), so use the literal.
            let bufSize = 4096
            var buf = [Int8](repeating: 0, count: bufSize)
            let ret = proc_pidpath(pid, &buf, UInt32(bufSize))
            let livePath: String?
            if ret > 0 {
                livePath = String(cString: buf)
            } else {
                // proc_pidpath failed (sandbox, race, etc.) — fail safe: do not kill.
                print("[RunRegistry] proc_pidpath failed for pid \(pid), skipping kill (safe)")
                try? fm.removeItem(at: fileURL)
                continue
            }

            guard livePath == entry.binPath else {
                // PID was recycled by a different process — do NOT kill anything.
                print("[RunRegistry] pid \(pid) recycled: live=\(livePath ?? "nil") expected=\(entry.binPath), skipping kill")
                try? fm.removeItem(at: fileURL)
                continue
            }

            // Exe-path matches — this is genuinely our orphaned trainer. Terminate it.
            print("[RunRegistry] reaping orphaned trainer pid \(pid) (\(entry.kind))")
            kill(pid, SIGTERM)

            // Poll briefly (up to 1 s in 100 ms increments) for graceful exit before SIGKILL.
            let deadline = Date().addingTimeInterval(1.0)
            while Date() < deadline {
                Thread.sleep(forTimeInterval: 0.1)
                if kill(pid, 0) != 0 { break } // gone
            }
            if kill(pid, 0) == 0 {
                // Still alive after grace period.
                print("[RunRegistry] pid \(pid) did not exit after SIGTERM, sending SIGKILL")
                kill(pid, SIGKILL)
            }

            try? fm.removeItem(at: fileURL)
            killed += 1
        }

        return killed
    }
}
