import os


def f_load():
    try:
        import dataclasses
        import math
        import subprocess

        import erutils
        import flet as ft
        import torch.cuda

        from baseline import generate
        from engine import config_model
    except ModuleNotFoundError as err:
        print(err)
        module = f"{err}".replace('No module named \'', '')[:-1]
        print(f'Downloading Module : {module}')
        ot = subprocess.run(f'pip install {module}')
        f_load()
    except FileNotFoundError as err:
        print(err)
        module = f"{err}".replace('No module named \'', '')[:-1]
        print(f'Downloading Module : {module}')
        ot = subprocess.run(f'pip install {module}')
        f_load()


f_load()

import dataclasses
import math

import flet as ft
import torch.cuda

from baseline import generate

DEBUG = False

from engine import config_model

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
class Cache:
    def __init__(self):
        self.GENERATOR_CONFIG = None
        self.model_path = None
        self.default_dtype = torch.float32
        self.default_device = 'cpu'

    ...


def main(page: ft.Page):
    cache = Cache()
    img_hw: int = 680
    page.window_min_width = 1760
    page.window_min_height = 920
    page.window_resizable = True
    page.theme_mode = ft.ThemeMode.SYSTEM
    # page.window_skip_task_bar = True

    page.title = "Creative Gan"
    cache.model_path = None
    default_dtype = torch.float32
    default_device = 'cpu'
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.update()
    cache.selected_option = options[0]
    fixed_left_margin = ft.margin.only(25, 0, 0, 0)
    field = ft.TextField(
        height=80,
        width=400,
        hint_text='Generate ...',
        border_radius=15

    )
    # Image Container
    image = ft.Container(border_radius=25, image_src=f"", height=img_hw, width=img_hw,
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
                         image_fit=ft.ImageFit.COVER, margin=ft.margin.only(top=80))

    # Check box items
    check_box_items = [
        ft.Container(ft.Switch(label=f'{option}', value=False, active_color='#010101', thumb_color='#2596be',
                               tooltip=f'{option} is an Option to use for generating Method ',
                               inactive_track_color='#d3df4', label_position=ft.LabelPosition.RIGHT),
                     margin=ft.margin.only(5, 8, 0, 0)) for option
        in options]
    check_box = ft.Container(
        ft.Column(
            check_box_items
        ),
        height=0,
        width=0,
        animate_size=ft.animation.Animation(800, ft.AnimationCurve.EASE_IN_OUT_CUBIC_EMPHASIZED),
        gradient=ft.LinearGradient(
            begin=ft.alignment.top_right,
            end=ft.alignment.Alignment(0.8, 1),
            colors=[
                "#2596be",
                "#ffffff",
            ],
            tile_mode=ft.GradientTileMode.CLAMP,
            rotation=math.pi / 3,
        ),
        border_radius=25,
        margin=fixed_left_margin
    )
    min_res = 512
    max_res = 2048
    cache.res = min_res

    def res_slider_on_change(e):
        cache.res = int(e.control.value)
        model_info_box_items[-2].value = f'Resolution : {cache.res} x {cache.res}'
        page.update()

    model_info_box_items = [
        ft.Container(
            ft.Dropdown(
                width=150,
                height=75,
                options=[
                    ft.dropdown.Option("Float 16"),
                    ft.dropdown.Option("Float 32"),
                    ft.dropdown.Option("BFloat 16"),
                ],
                tooltip='define the model data type recommended (Float 32)',
                border_radius=25,
                label='Model Data Type'
            )
        ),
        ft.Container(
            ft.Dropdown(
                width=150,
                height=75,
                options=[
                    ft.dropdown.Option("cpu"),
                    ft.dropdown.Option("cuda"),
                ],
                tooltip='define the model Device',
                border_radius=25,
                label='Model Device'
            )
        ),
        ft.Text(f'Resolution : {cache.res} x {cache.res}'),
        ft.Container(
            ft.Slider(
                min=min_res, max=max_res, divisions=(max_res - min_res) // 8, label='{value} x {value}',
                width=200, height=10,
                on_change=res_slider_on_change
            )
        )

    ]

    model_info_box = ft.Container(
        ft.Column(
            model_info_box_items,
            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        ),
        height=0,
        width=0,
        animate_size=ft.animation.Animation(800, ft.AnimationCurve.EASE_IN_OUT_CUBIC_EMPHASIZED),
        gradient=ft.LinearGradient(
            begin=ft.alignment.top_right,
            end=ft.alignment.Alignment(0.8, 1),
            colors=[
                "#2596be",
                "#ffffff",
            ],
            tile_mode=ft.GradientTileMode.CLAMP,
            rotation=math.pi / 3,
        ),
        border_radius=25,
        margin=fixed_left_margin
    )

    def close_banner(e=None):
        page.banner.open = False
        page.update()

    def show_banner_click(e=None):
        page.banner.open = True
        page.update()

    page.banner = ft.Banner(

        bgcolor='#0095cf',
        leading=ft.Icon(ft.icons.WARNING_SHARP, color='#17395f', size=40),
        content=ft.Text(
            "Oops, there were some errors while trying to generate image make"
            " sure that you haven\'t pass empty field and model is also loaded"
        ),
        actions=[
            ft.TextButton("Ok !", on_click=close_banner),
            ft.TextButton("Field wasn't empty !", on_click=close_banner),

        ],
    )

    def close_dlg_out_memory(e):
        dlg_modal_out_memory.open = False
        page.update()

    def close_dlg_model_not_selected(e):
        dlg_modal_model_not_selected.open = False
        page.update()

    dlg_modal_out_memory = ft.AlertDialog(
        modal=True,
        title=ft.Text("Error"),
        content=ft.Text("You Got cuda out of memory error try to use cpu"),
        actions=[
            ft.TextButton("Ok", on_click=close_dlg_out_memory),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )

    dlg_modal_model_not_selected = ft.AlertDialog(
        modal=True,
        title=ft.Text("Error"),
        content=ft.Text("You should select model path to load first"),
        actions=[
            ft.TextButton("Ok", on_click=close_dlg_model_not_selected),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )

    def open_dlg_modal_out_memory(e):
        page.dialog = dlg_modal_out_memory
        dlg_modal_out_memory.open = True
        page.update()

    def open_dlg_modal_model_not_selected(e):
        page.dialog = dlg_modal_model_not_selected
        dlg_modal_model_not_selected.open = True
        page.update()

    def file_picker(e: ft.FilePickerResultEvent):
        path_to_json = ''.join(f + os.sep for f in e.files[0].path.split(os.sep)[:-1])
        cache.model_path = path_to_json
        #

    def load_model_(e):
        model_info_box.height = 0
        model_info_box.width = 0
        page.update()
        if cache.model_path is not None:
            col1_items[0].controls[2].content.disabled = True
            col1_items[0].controls[1].content.disabled = True
            page.update()

            cache.dtype = model_info_box_items[0].content.value if model_info_box_items[
                                                                       0].content.value is not None else default_dtype
            cache.device = model_info_box_items[1].content.value if model_info_box_items[
                                                                        1].content.value is not None else default_device
            print(cache.device)
            print(cache.dtype)
            try:
                cache.model_ckpt = config_model(model_path=r'{}'.format(cache.model_path), nsfw_allowed=True,
                                                device=cache.device)

                cache.GENERATOR_CONFIG = dict(
                    model=cache.model_ckpt,
                    out_dir='tools/assets',
                    use_version=True, version='v4',
                    use_realistic=False, image_format='png',
                    nsfw_allowed=True,
                    use_check_prompt=False, task='save'
                )
            except torch.cuda.OutOfMemoryError as err:
                col1_items[0].controls[2].content.disabled = False
                col1_items[0].controls[1].content.disabled = False
                open_dlg_modal_out_memory(None)
        else:
            open_dlg_modal_model_not_selected(None)

    def add_clicked(e):
        max_c_width = 300
        max_c_height = 85
        progress_bar_inited = ft.ProgressBar(width=max_c_width, bar_height=max_c_height // 6, bgcolor='#f0f6f7ff',
                                             color='#2596be', )

        def update_progress_bar(prompt_):
            try:
                for i in generate(prompt=prompt_, size=(cache.res, cache.res),
                                  **cache.GENERATOR_CONFIG):
                    cnt = int(i) * 2
                    cnt = cnt if cnt != 0 else 2
                    progress_bar_inited.value = cnt * 0.01
                    page.update()
            except torch.cuda.OutOfMemoryError as err:
                col1_items[0].controls[2].content.disabled = False
                col1_items[0].controls[1].content.disabled = False
                open_dlg_modal_out_memory(None)

        if field.value != '' and hasattr(cache, 'model_ckpt'):
            col1_items.append(
                ft.Container(
                    ft.Column(
                        [
                            ft.Text('Generating ' + field.value, style=ft.TextThemeStyle.HEADLINE_SMALL,
                                    ),
                            progress_bar_inited
                        ],
                        alignment=ft.MainAxisAlignment.START,
                        horizontal_alignment=ft.CrossAxisAlignment.START
                    )
                    , width=max_c_width, height=max_c_height, margin=fixed_left_margin

                )
            )
            prompt = field.value
            for t in check_box_items:
                prompt += ', ' + t.content.label if t.content.value else ''

            field.value = ""
            update_progress_bar(prompt_=prompt)
            image.image_src = f'{prompt}.png'
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

        check_box.height = 0 if not vanished else len(check_box_items) * 60
        check_box.width = 0 if not vanished else 300
        page.update()

    def show_model_info(e):
        vanished = True if model_info_box.width == 0 else False
        model_info_box.height = 0 if not vanished else 5 * 60
        model_info_box.width = 0 if not vanished else 300
        page.update()

    file_picker_object = ft.FilePicker(on_result=file_picker)
    page.overlay.append(file_picker_object)
    page.update()
    # def load_file_picker(e):
    #     file_picker_object.pick_files()
    col1_items = [
        ft.Row([
            ft.Container(ft.ElevatedButton(
                "Options",
                on_click=show_options,
            ), margin=fixed_left_margin),
            ft.Container(ft.ElevatedButton(
                "Model Info",
                on_click=show_model_info,
                disabled=False,
            ), margin=fixed_left_margin),
            ft.Container(ft.ElevatedButton(
                "Load Model",
                on_click=load_model_,
                disabled=False,
            ), margin=fixed_left_margin),
            ft.Container(ft.ElevatedButton(
                "Load Model File",
                on_click=lambda _: file_picker_object.pick_files(allowed_extensions=['json', 'yaml', 'erutils']),
                disabled=False,
            ), margin=fixed_left_margin)
        ]),

        check_box, model_info_box,
        main_column,
        tool_tip,

    ]

    def theme_changed(e):
        page.theme_mode = (
            ft.ThemeMode.DARK
            if page.theme_mode == ft.ThemeMode.LIGHT
            else ft.ThemeMode.LIGHT
        )
        theme_changer.label = (
            "Light theme" if page.theme_mode == ft.ThemeMode.LIGHT else "Dark theme"
        )
        colors = [
            "#2596be",
            "#ffffff" if page.theme_mode == ft.ThemeMode.LIGHT else '#000000',
        ]
        check_box.gradient.colors = colors
        model_info_box.gradient.colors = colors
        page.update()

    page.theme_mode = ft.ThemeMode.LIGHT
    theme_changer = ft.Switch(label="Light theme", on_change=theme_changed)
    # Main Display
    col1 = ft.Column(
        col1_items,
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.START
    )

    col2 = ft.Column(
        [
            image, theme_changer
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
