# ADR-087 Amendment 1 — open pull request population snapshot

Immutable record of the population the amendment's join was measured over.
Counts in the amendment are derived from this file, not from a live query.

- Cutoff (UTC): 2026-08-04T08:26:43+00:00
- Surface ref: `e842a892d950f3d3b2687eaf6cc573e0587785a7`
- Open pull requests: **35**
- Trigger the rule (touch `crates/inference/`, `crates/embed/` or `crates/fann/`): **26**
- Reached by the default A/B surface: **1**

Names are classified on both `filename` and `previous_filename`, so a rename is
classified at both of its paths.

| PR    | head                                       | draft | files | triggers | reached paths             |
| ----- | ------------------------------------------ | ----- | ----- | -------- | ------------------------- |
| #1021 | `fa979d957f81f91d94db99eca17ad6d2bf9385ef` | yes   | 21    | yes      | —                         |
| #1037 | `b3fdf0e3526fa596009a5c3fce4ab427891c508b` | yes   | 11    | yes      | —                         |
| #1039 | `032f218881f50e8da1391f6d9c837cd5831943ae` | yes   | 10    | no       | —                         |
| #1150 | `5ff852251c0179fea47b9685b0d440a415ad8404` | yes   | 5     | yes      | —                         |
| #1162 | `c976e1215e68f7dac82cccce5adca7c1f7bb4308` | yes   | 9     | yes      | —                         |
| #1183 | `c249e1d0cbefccbfee69253233186397fde5d1da` | no    | 3     | yes      | —                         |
| #1187 | `37170b4455d8077d8f761ffc443f5443b871fa84` | yes   | 1     | yes      | —                         |
| #1208 | `929ac7c620857045f883a3cc0d2de7c59be8c30e` | yes   | 6     | yes      | —                         |
| #1219 | `a888bf0ddef1d5624f83bcdf0943a74c7eb154f6` | no    | 8     | yes      | —                         |
| #1230 | `852649a87248195dccf0fb63df0e4a62e3ea9ea8` | no    | 13    | yes      | —                         |
| #1231 | `228417020736ae5b3a46aff6954b4ba7fb064f04` | yes   | 20    | yes      | —                         |
| #1241 | `2f9950cd144c8e57f68a012fbc120d38b648988e` | no    | 4     | yes      | —                         |
| #1244 | `1e8ba317241d7b7dc674516a67d0ef10c5dcfe03` | yes   | 2     | yes      | —                         |
| #1245 | `a80a93a4b0d3f7cf3abcab7a2f3d3ef35e799c93` | yes   | 2     | yes      | —                         |
| #1246 | `e6141e58d3c54b6b8625f4346ec967dbb5c79a42` | yes   | 7     | yes      | —                         |
| #1247 | `412d00a3adb5d6e17468a304f83e8a16752569ca` | yes   | 7     | yes      | —                         |
| #1253 | `42bcb2299e2e4a7c818c15ff9b5bfc4fbe9bd858` | yes   | 10    | yes      | —                         |
| #1255 | `8de086f1314a25c84018a43a475deb8a352f05f7` | yes   | 5     | yes      | —                         |
| #1256 | `2a92783e5d1f5fb4ea9e5095da24b422b1cb17c7` | yes   | 24    | yes      | —                         |
| #1260 | `b372458dc5fb0689d583d392dee1a706ab8beddb` | yes   | 17    | no       | —                         |
| #1262 | `a1a6e8eeb0f4676b6abd82cc995757aed91fea4f` | yes   | 4     | yes      | —                         |
| #1263 | `16db5498945b9d57f55f8faf0330f87dd198a046` | yes   | 19    | yes      | —                         |
| #1264 | `ad2a003d55506d396ffbfb9fca392ad4126ce550` | no    | 5     | yes      | —                         |
| #1265 | `5a48400bd1f33a6259c23c474fa2cb0683fbad95` | no    | 3     | yes      | —                         |
| #1266 | `a25e9053dc0f8a37f315691168a922e9bead2f70` | no    | 3     | yes      | —                         |
| #1267 | `92a60a5b5507b29aebb613d6e790918593b56548` | yes   | 6     | yes      | —                         |
| #1270 | `ae6d27b5a61d2a62f912d195a644b1837a893fab` | yes   | 20    | yes      | —                         |
| #1289 | `1a77f234ad38edba155537cf7c3c375e0496dcc2` | yes   | 18    | yes      | `crates/embed/src/lib.rs` |
| #1290 | `dcd4896b96b3eeadfb6afdf1e6b72b049fce5595` | no    | 1     | no       | —                         |
| #1296 | `1464edf523d1de3667932fa019082337fc8fd3aa` | no    | 1     | no       | —                         |
| #1297 | `ff38dba96fe383827fd4865898c298c5f50e6cf1` | no    | 2     | no       | —                         |
| #1298 | `2dab9e7d2cd17a7bb5010d029bd91c3694dcecb9` | no    | 3     | no       | —                         |
| #1303 | `61049c7443526418f7537f97f64eaa6ff251373d` | no    | 1     | no       | —                         |
| #1304 | `9d521449325ab76a3bef7cf6dac4665001c358ca` | no    | 2     | no       | —                         |
| #1306 | `cd6d65c753c85d28bcadb42d061988bf3a0f8b0e` | no    | 1     | no       | —                         |
