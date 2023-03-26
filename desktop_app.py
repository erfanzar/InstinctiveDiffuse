import dataclasses
import math
import time

import flet as ft

DEBUG = True
if not DEBUG:
    generator_kwargs = dict(use_version=True, version='v4', use_realistic=False, size=size, nsfw_allowed=nsfw_allowed,
                            out_dir=out_dir)
    device = 'cuda'
    grc = config_model(model_path=r'{}'.format(opt.model_path), nsfw_allowed=True, device=device)

options = [
    'real',
    'realistic',
    'smooth',
    'sharp',
    'realistic,smooth',
    '8k',
    'detailed',
    'smart'
]


@dataclasses.dataclass
class Cash:
    ...


def main(page: ft.Page):
    cash = Cash()
    img_hw: int = 680
    page.window_width = 1760
    page.window_height = 920
    page.window_resizable = False
    page.theme_mode = 'light'
    page.title = "Creative Gan"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.update()
    cash.selected_option = options[0]
    fixed_left_margin = ft.margin.only(25, 0, 0, 0)
    field = ft.TextField(
        height=80,
        width=400,
        hint_text='Generate ...',
        border_radius=15

    )
    # Image Container
    image = ft.Container(border_radius=5, bgcolor=ft.colors.BLUE, image_src=f"/ALMT.png", height=img_hw, width=img_hw,
                         image_fit=ft.ImageFit.COVER, margin=ft.margin.only(top=80))

    # Check box items
    check_box_items = [
        ft.Switch(label=f'{option}', value=False, active_color=ft.colors.CYAN, thumb_color=ft.colors.CYAN,
                  tooltip=f'{option} is an Option to use for generating Method ') for option
        in options]
    check_box = ft.Container(ft.Column(
        check_box_items
    ), height=0, width=0, animate_size=ft.animation.Animation(400, ft.AnimationCurve.EASE_IN_TO_LINEAR),
        bgcolor=ft.colors.CYAN_50, border_radius=35)

    def close_banner(e=None):
        page.banner.open = False
        page.update()

    def show_banner_click(e=None):
        page.banner.open = True
        page.update()

    page.banner = ft.Banner(
        bgcolor=ft.colors.AMBER_100,
        leading=ft.Icon(ft.icons.WARNING_AMBER_ROUNDED, color=ft.colors.AMBER, size=40),
        content=ft.Text(
            "Oops, there were some errors while trying to generate image make sure that you haven\'t pass empty field"
        ),
        actions=[
            ft.TextButton("Ok !", on_click=close_banner),
            ft.TextButton("Field wasn't empty !", on_click=close_banner),

        ],
    )

    def add_clicked(e):
        max_c_width = 300
        max_c_height = 85
        progress_bar_inited = ft.ProgressBar(width=max_c_width, bar_height=max_c_height // 5, bgcolor=ft.colors.BLACK,
                                             color='#e0e0e0', )

        def update_progress_bar():

            for i in range(0, 101):
                progress_bar_inited.value = i * 0.01
                time.sleep(0.02)
                page.update()

        if field.value != '':
            col1_items.append(
                ft.Container(
                    ft.Column(
                        [
                            ft.Text('Generating ' + field.value, style=ft.TextThemeStyle.HEADLINE_MEDIUM,
                                    ),
                            progress_bar_inited
                        ],
                        alignment=ft.MainAxisAlignment.START,
                        horizontal_alignment=ft.CrossAxisAlignment.START
                    )
                    , width=max_c_width, height=max_c_height, margin=fixed_left_margin

                )
            )

            field.value = ""
            update_progress_bar()
            image.image_src = 'GC.png'
            page.update()
        else:
            show_banner_click()

    # ToolTIP
    tool_tip = ft.Container(ft.Tooltip(
        message="Using Creative Gan is easiest possible thing just write the text in Text field and get image Done !",
        content=ft.Text("TOOLTIP ?", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
        padding=20,
        border_radius=10,
        text_style=ft.TextStyle(size=20, color=ft.colors.WHITE),
        gradient=ft.LinearGradient(
            begin=ft.alignment.top_left,
            end=ft.alignment.Alignment(0.8, 1),
            colors=[
                "0xff1f005c",
                "0xff5b0060",
                "0xff870160",
                "0xffac255e",
                "0xffca485c",
                "0xffe16b5c",
                "0xfff39060",
                "0xffffb56b",
            ],
            tile_mode=ft.GradientTileMode.MIRROR,
            rotation=math.pi / 3,
        ),
    ), margin=fixed_left_margin)
    main_column = ft.Column(
        controls=[
            ft.Row(
                [
                    ft.Container(field, margin=ft.margin.only(25, 15, 0, 0)),
                    ft.Container(ft.FloatingActionButton(icon=ft.icons.ADD, on_click=add_clicked, height=65, width=65),
                                 margin=ft.margin.only(0, 5, 0, 0))
                ],
                alignment=ft.MainAxisAlignment.START
            )
        ]
    )

    def show_options(e):
        vanished = True if check_box.width == 0 else False

        check_box.height = 0 if not vanished else 450
        check_box.width = 0 if not vanished else 300
        page.update()

    col1_items = [
        ft.Container(ft.ElevatedButton(
            "Options",
            on_click=show_options,
        ), margin=fixed_left_margin),
        check_box,
        main_column,
        tool_tip,

    ]
    # Main Display
    col1 = ft.Column(
        col1_items,
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.START
    )
    col2 = ft.Column(
        [
            image
        ]
    )
    page.add(
        ft.Row(
            [
                col1,
                col2,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER
        )
    )


if __name__ == "__main__":
    ft.app(target=main, assets_dir='tools/assets', )
