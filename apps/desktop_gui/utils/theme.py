import dearpygui.dearpygui as dpg


def apply_global_theme() -> None:
    """Apply a modern dark theme with refined color palette and styling."""

    bg_primary = (13, 17, 23)
    bg_secondary = (20, 27, 38)
    bg_tertiary = (26, 35, 50)
    border_subtle = (40, 52, 72)
    border_accent = (59, 85, 130)

    text_primary = (234, 240, 250)
    text_secondary = (160, 175, 205)
    text_disabled = (100, 115, 140)

    accent_blue = (59, 130, 246)
    accent_blue_hover = (96, 165, 250)
    accent_blue_active = (37, 99, 235)
    accent_teal = (45, 212, 191)
    accent_green = (34, 197, 94)
    accent_green_hover = (52, 211, 153)
    accent_green_active = (5, 150, 105)
    accent_red = (239, 68, 68)
    accent_red_hover = (248, 113, 113)
    accent_red_active = (220, 38, 38)
    accent_purple = (167, 139, 250)
    accent_orange = (251, 191, 36)
    accent_cyan = (103, 232, 249)

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, bg_primary)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, bg_secondary)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, bg_secondary)
            dpg.add_theme_color(dpg.mvThemeCol_Border, border_subtle)
            dpg.add_theme_color(dpg.mvThemeCol_Text, text_primary)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, text_disabled)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (17, 24, 36))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, bg_tertiary)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (32, 45, 65))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 50, 80))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (40, 65, 100))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 75, 115))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (25, 35, 52))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (35, 48, 70))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (45, 60, 85))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, accent_green)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, accent_blue)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, accent_blue_hover)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, accent_blue)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, bg_secondary)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, bg_tertiary)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, bg_secondary)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, bg_secondary)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, border_accent)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, accent_blue)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, accent_blue_active)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, border_subtle)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, border_accent)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, accent_blue)

            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 14)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 16)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 12, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 12, 10)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 12)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_SeparatorTextPadding, 8, 4)

    dpg.bind_theme(global_theme)

    with dpg.theme(tag="theme_button_primary"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, accent_blue)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, accent_blue_hover)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, accent_blue_active)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 16, 8)

    with dpg.theme(tag="theme_button_success"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, accent_green)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, accent_green_hover)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, accent_green_active)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 16, 8)

    with dpg.theme(tag="theme_button_danger"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, accent_red)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, accent_red_hover)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, accent_red_active)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 16, 8)

    with dpg.theme(tag="theme_button_subtle"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (36, 39, 48))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (48, 53, 66))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (58, 66, 81))
            dpg.add_theme_color(dpg.mvThemeCol_Text, text_primary)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 14, 8)

    with dpg.theme(tag="theme_text_accent"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, accent_teal)

    with dpg.theme(tag="theme_text_success"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, accent_green)

    with dpg.theme(tag="theme_text_warning"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, accent_orange)

    with dpg.theme(tag="theme_text_error"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, accent_red)

    with dpg.theme(tag="theme_input"):
        with dpg.theme_component(dpg.mvInputText):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (17, 24, 36))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, bg_tertiary)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (32, 45, 65))
            dpg.add_theme_color(dpg.mvThemeCol_Text, text_primary)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, text_disabled)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 12, 8)

    with dpg.theme(tag="theme_slider"):
        with dpg.theme_component(dpg.mvSliderInt):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, bg_secondary)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, bg_tertiary)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (32, 45, 65))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, accent_blue)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, accent_blue_active)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)

    with dpg.theme(tag="theme_separator"):
        with dpg.theme_component(dpg.mvSeparator):
            dpg.add_theme_color(dpg.mvThemeCol_Separator, border_subtle)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, border_accent)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, accent_blue)

    with dpg.theme(tag="theme_checkbox"):
        with dpg.theme_component(dpg.mvCheckbox):
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, accent_green)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, bg_secondary)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, bg_tertiary)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (32, 45, 65))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)

    with dpg.theme(tag="theme_progress_bar"):
        with dpg.theme_component(dpg.mvProgressBar):
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, accent_blue)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)

    with dpg.theme(tag="theme_card"):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (25, 27, 33))
            dpg.add_theme_color(dpg.mvThemeCol_Border, border_subtle)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 16)

    with dpg.theme(tag="theme_card_soft"):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (17, 20, 28))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (48, 54, 68))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 14)
