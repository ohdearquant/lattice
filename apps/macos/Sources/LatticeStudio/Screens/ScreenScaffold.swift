import SwiftUI

// Shared chrome for every screen: a ruled header (index · title + subtitle), an
// optional trailing slot for a primary action / live status, then the content body
// on the instrument canvas. Screens compose their own layouts inside `content`.
struct ScreenScaffold<Trailing: View, Content: View>: View {
    let screen: Screen
    let subtitle: String
    @ViewBuilder var trailing: () -> Trailing
    @ViewBuilder var content: () -> Content

    init(
        screen: Screen,
        subtitle: String,
        @ViewBuilder trailing: @escaping () -> Trailing = { EmptyView() },
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.screen = screen
        self.subtitle = subtitle
        self.trailing = trailing
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.lg) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(screen.title.localizedCapitalized)
                        .font(Theme.Fonts.display(22, .bold))
                        .foregroundStyle(Theme.Palette.ink)
                    Text(subtitle)
                        .font(Theme.Fonts.body)
                        .foregroundStyle(Theme.Palette.textSecondary)
                }
                Spacer(minLength: Theme.Space.lg)
                trailing()
            }
            content()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(Theme.Space.xl)
    }
}
