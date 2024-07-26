import ipywidgets as widgets
from ipywidgets import Layout, HBox, VBox
from IPython.display import display, clear_output
if 'google.colab' in str(get_ipython()):
    from MPEAsAlloyPublic.optimiser import *
else:
    from optimiser import *


def extractSettingsFromGUI(GUI_inputs, mode):
    settings = scanSettings(mode)

    for key in settings.concentration_inputs:
        settings.concentration_inputs[key] = [GUI_inputs['concentration_inputs'][key][0].value]


    for key in settings.categorical_inputs:
        settings.categorical_inputs[key] = []
        for index, value in enumerate(settings.categorical_inputs_info[key]['tag']):
            if GUI_inputs['categorical_inputs'][key][0].value==value:
                settings.categorical_inputs[key].append(1)
            else:
                settings.categorical_inputs[key].append(0)

    for key in settings.range_based_inputs:
        settings.range_based_inputs[key] = [GUI_inputs['range_based_inputs'][key][0].value]


    return settings

def generateModeSelectionGUI(mode = 'Corrosion'):
    mode_dropdown = widgets.Dropdown(
        options=['Corrosion'],
        value=mode,
        description='<b>Select Mode:</b>',
        style={'description_width': 'initial'},
        disabled=False
    )
    display(mode_dropdown)
    generateMainGUI(mode)
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            clear_output(wait=True)
            generateModeSelectionGUI(change['new'])
    mode_dropdown.observe(on_change)

def generateMainGUI(mode):
    settings = scanSettings(mode)
    KEY_LABEL_WIDTH = "30px"
    TO_LABEL_WIDTH = "15px"
    INPUT_BOX_WIDTH = "70px"
    INPUT_BOX_HEIGHT = "20px"

    LEFT_RIGHT_PADDING = Layout(margin="0px 100px 0px 100px")
    BOTTOM_PADDING = Layout(margin="0px 0px 5px 0px")

    default_input_box_layout = Layout(width=INPUT_BOX_WIDTH, height=INPUT_BOX_HEIGHT)

    GUI_inputs = {"concentration_inputs" :{},
                  "categorical_inputs": {},
                  "range_based_inputs": {}
                  }

    range_based_inputs_VBox = [widgets.HTML("<b>Compositional range (wt. %)    </b>")]
    for key in settings.range_based_inputs:
        key_label = widgets.Label(f"{key}:", layout=Layout(width=KEY_LABEL_WIDTH))
        lower_bound_box = widgets.FloatText(value=settings.range_based_inputs[key][0], layout=default_input_box_layout)
        range_based_inputs_VBox.append(HBox([key_label, lower_bound_box]))
        GUI_inputs["range_based_inputs"][key] = [lower_bound_box]


    concentration_inputs_VBox = [widgets.HTML("<b>Electrolyte concentration (M)    </b>")]
    for key in settings.concentration_inputs:
        key_label = widgets.Label(f"{key}:", layout=Layout(width="120px"))
        lower_bound_box = widgets.FloatText(value=settings.concentration_inputs[key][0], layout=default_input_box_layout)
        concentration_inputs_VBox.append(HBox([key_label, lower_bound_box]))
        GUI_inputs["concentration_inputs"][key] = [lower_bound_box]

        

    categorical_inputs_VBox = [widgets.HTML("<b>  </b>")]
    for key in settings.categorical_inputs:
        categorical_inputs_VBox.append(widgets.HTML(f'{key}:'))
        GUI_inputs["categorical_inputs"][key] = []
        options = []
        for i, value in enumerate(settings.categorical_inputs_info[key]['tag']):
            options.append(value)
        value_checkbox = widgets.RadioButtons(options=options,
                                             description = '',
                                              disabled=False,
                                              indent=False)
        
#             if value in settings.categorical_inputs[key]:
#                 value_checkbox.value = True
        categorical_inputs_VBox.append(value_checkbox)
        GUI_inputs["categorical_inputs"][key].append(value_checkbox)

        

    first_column = VBox(range_based_inputs_VBox)
    

    second_column = VBox([VBox(categorical_inputs_VBox, layout =LEFT_RIGHT_PADDING )])

    third_column = VBox(concentration_inputs_VBox, layout=BOTTOM_PADDING)

    
    display(HBox([first_column, second_column, third_column]))

    run_scan_button = widgets.Button(description="Run")
    display(run_scan_button)

    def on_button_clicked(b):
        print()
        print('start running model ....')
        optimiser(extractSettingsFromGUI(GUI_inputs, mode))

    run_scan_button.on_click(on_button_clicked)
