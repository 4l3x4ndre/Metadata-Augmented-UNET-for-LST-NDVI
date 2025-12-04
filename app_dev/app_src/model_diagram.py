import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState



def instantiate_unet_diagram(metadata_length:int=4):
    if metadata_length == 4:
        metadata = '(lat, lon, population, delta)'
    elif metadata_length == 8:
        metadata = '(lat, lon, population, delta, year1, month1, year2, month2)'
    else:
        metadata = 'Metadata'

    # Define the nodes
    nodes = [
        StreamlitFlowNode(id='01', pos=(-200, 600), data={'content': metadata}, node_type='input', source_position='right', draggable=False),
        StreamlitFlowNode(id='02', pos=(-200, 700), data={'content': 'Temperature history'}, node_type='input', source_position='right', draggable=False),
        StreamlitFlowNode(id='1', pos=(-200, 200), data={'content': 'Spatial Input Stack'}, node_type='input', source_position='right', draggable=False),
        StreamlitFlowNode(id='3', pos=(50, 600), data={'content': 'Metadata Encoder MLP'}, node_type='default', source_position='right', draggable=False),
        StreamlitFlowNode(id='2', pos=(50, 700), data={'content': 'Temporal Encoder LSTM'}, node_type='default', source_position='right', draggable=False),
        StreamlitFlowNode(id='4', pos=(50, 200), data={'content': 'conv0_0'}, node_type='default', source_position='right', draggable=False),
        StreamlitFlowNode(id='5', pos=(50, 300), data={'content': 'conv1_0'}, node_type='default', source_position='right', draggable=False),
        StreamlitFlowNode(id='6', pos=(50, 400), data={'content': 'conv2_0'}, node_type='default', source_position='right', draggable=False),
        StreamlitFlowNode(id='7', pos=(50, 500), data={'content': 'conv3_0'}, node_type='default', source_position='right', draggable=False),
        StreamlitFlowNode(id='8', pos=(300, 600), data={'content': 'Bottleneck'}, node_type='default', source_position='right', draggable=False),
        StreamlitFlowNode(id='9', pos=(550, 500), data={'content': 'conv3_1'}, node_type='default', source_position='right', target_position='left', draggable=False),
        StreamlitFlowNode(id='10', pos=(550, 400), data={'content': 'conv2_1'}, node_type='default', source_position='right', target_position='left', draggable=False),
        StreamlitFlowNode(id='11', pos=(550, 300), data={'content': 'conv1_1'}, node_type='default', source_position='right', target_position='left', draggable=False),
        StreamlitFlowNode(id='12', pos=(550, 200), data={'content': 'conv0_1'}, node_type='default', source_position='right', target_position='left', draggable=False),
        StreamlitFlowNode(id='13', pos=(800, 200), data={'content': 'Final Conv'}, node_type='default', source_position='right', target_position='left', draggable=False),
        StreamlitFlowNode(id='14', pos=(900, 200), data={'content': 'Output'}, node_type='output', target_position='left', draggable=False),
    ]

    # Define the edges
    edges = [
        StreamlitFlowEdge('01-3', '01', '3', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('02-2', '02', '2', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('1-4', '1', '4', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('4-5', '4', '5', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('5-6', '5', '6', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('6-7', '6', '7', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('7-8', '7', '8', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('2-8', '2', '8', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('3-8', '3', '8', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('8-9', '8', '9', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('7-9', '7', '9', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('9-10', '9', '10', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('6-10', '6', '10', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('10-11', '10', '11', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('5-11', '5', '11', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('11-12', '11', '12', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('4-12', '4', '12', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('12-13', '12', '13', animated=True, marker_end={'type': 'arrow'}),
        StreamlitFlowEdge('13-14', '13', '14', animated=True, marker_end={'type': 'arrow'}),
    ]

    # Initialize the flow state
    if 'static_flow_state' not in st.session_state:
        st.session_state.static_flow_state = StreamlitFlowState(nodes, edges)

    # Display the flow diagram
    streamlit_flow('static_flow',
                   st.session_state.static_flow_state,
                   fit_view=True,
                   show_minimap=False,
                   show_controls=False,
                   pan_on_drag=True,
                   allow_zoom=True)



def instantiate_unetpp_diagram(metadata_length:int=4):
    if metadata_length == 4:
        metadata = '(lat, lon, population, delta)'
    elif metadata_length == 8:
        metadata = '(lat, lon, population, delta, year1, month1, year2, month2)'
    else:
        metadata = 'Metadata'

    edge_style = {
        'stroke': '#3d73c4',
    }
    black_edge_style = {
        'stroke': 'black',
    }

    # Node positions for pyramid
    decoder_y = [250, 400, 550, 700]
    decoder_y2 = [250, 400, 550]
    decoder_y3 = [325, 550]
    decoder_y4 = [400]

    nodes = [
        StreamlitFlowNode(id='01', pos=(-500, 650), data={'content': metadata}, node_type='input', source_position='right', draggable=True),
        StreamlitFlowNode(id='02', pos=(-500, 850), data={'content': 'Temperature history'}, node_type='input', source_position='right', draggable=True),
        StreamlitFlowNode(id='1', pos=(-500, 250), data={'content': 'Spatial Input Stack'}, node_type='input', source_position='right', draggable=True),

        StreamlitFlowNode(id='3', pos=(-300, 650), data={'content': 'Metadata Encoder MLP'}, node_type='default', source_position='right', draggable=True),
        StreamlitFlowNode(id='2', pos=(-300, 850), data={'content': 'Temporal Encoder LSTM'}, node_type='default', source_position='right', draggable=True),

        StreamlitFlowNode(id='4', pos=(0, 250), data={'content': 'conv0_0 (Encoder)'}, node_type='default', source_position='right', draggable=True),
        StreamlitFlowNode(id='5', pos=(0, 400), data={'content': 'conv1_0 (Encoder)'}, node_type='default', source_position='right', draggable=True),
        StreamlitFlowNode(id='6', pos=(0, 550), data={'content': 'conv2_0 (Encoder)'}, node_type='default', source_position='right', draggable=True),
        StreamlitFlowNode(id='7', pos=(0, 700), data={'content': 'conv3_0 (Encoder)'}, node_type='default', source_position='right', draggable=True),
        StreamlitFlowNode(id='8', pos=(0, 850), data={'content': 'conv4_0 (Encoder, deepest)'}, node_type='default', source_position='right', draggable=True),

        # Decoder nodes with embedding fusion (pyramid, centered)
        StreamlitFlowNode(id='9', pos=(350, decoder_y[0]), data={'content': 'conv0_1 (Decoder, Embedding Fusion)'}, node_type='default', source_position='right', target_position='left', draggable=True),
        StreamlitFlowNode(id='10', pos=(350, decoder_y[1]), data={'content': 'conv1_1 (Decoder, Embedding Fusion)'}, node_type='default', source_position='right', target_position='left', draggable=True),
        StreamlitFlowNode(id='11', pos=(350, decoder_y[2]), data={'content': 'conv2_1 (Decoder, Embedding Fusion)'}, node_type='default', source_position='right', target_position='left', draggable=True),
        StreamlitFlowNode(id='12', pos=(350, decoder_y[3]), data={'content': 'conv3_1 (Decoder, Embedding Fusion)'}, node_type='default', source_position='right', target_position='left', draggable=True),

        StreamlitFlowNode(id='13', pos=(650, decoder_y2[0]), data={'content': 'conv0_2 (Decoder, Embedding Fusion)'}, node_type='default', source_position='right', target_position='left', draggable=True),
        StreamlitFlowNode(id='14', pos=(650, decoder_y2[1]), data={'content': 'conv1_2 (Decoder, Embedding Fusion)'}, node_type='default', source_position='right', target_position='left', draggable=True),
        StreamlitFlowNode(id='15', pos=(650, decoder_y2[2]), data={'content': 'conv2_2 (Decoder, Embedding Fusion)'}, node_type='default', source_position='right', target_position='left', draggable=True),

        StreamlitFlowNode(id='16', pos=(1000, decoder_y3[0]), data={'content': 'conv0_3 (Decoder, Embedding Fusion)'}, node_type='default', source_position='right', target_position='left', draggable=True),
        StreamlitFlowNode(id='17', pos=(1000, decoder_y3[1]), data={'content': 'conv1_3 (Decoder, Embedding Fusion)'}, node_type='default', source_position='right', target_position='left', draggable=True),

        StreamlitFlowNode(id='18', pos=(1300, decoder_y4[0]), data={'content': 'conv0_4 (Decoder, Embedding Fusion)'}, node_type='default', source_position='right', target_position='left', draggable=True),

        StreamlitFlowNode(id='19', pos=(1550, decoder_y4[0]), data={'content': 'Final Conv'}, node_type='default', source_position='right', target_position='left', draggable=True),
        StreamlitFlowNode(id='20', pos=(1660, decoder_y4[0]), data={'content': 'Output'}, node_type='output', target_position='left', draggable=True),
    ]

    edges = [
        # Inputs to encoders
        StreamlitFlowEdge('01-3', '01', '3', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('02-2', '02', '2', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('1-4', '1', '4', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('4-5', '4', '5', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('5-6', '5', '6', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('6-7', '6', '7', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('7-8', '7', '8', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),

        # Decoder pyramid, skip connections, and embedding fusion
        # Level 1
        StreamlitFlowEdge('4-9', '4', '9', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('5-9', '5', '9', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('2-9', '2', '9', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('3-9', '3', '9', animated=True, marker_end={'type': 'arrow'}, style=edge_style),

        # Level 2
        StreamlitFlowEdge('5-10', '5', '10', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('6-10', '6', '10', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('2-10', '2', '10', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('3-10', '3', '10', animated=True, marker_end={'type': 'arrow'}, style=edge_style),

        # Level 3
        StreamlitFlowEdge('6-11', '6', '11', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('7-11', '7', '11', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('2-11', '2', '11', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('3-11', '3', '11', animated=True, marker_end={'type': 'arrow'}, style=edge_style),

        # Level 4
        StreamlitFlowEdge('7-12', '7', '12', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('8-12', '8', '12', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('2-12', '2', '12', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('3-12', '3', '12', animated=True, marker_end={'type': 'arrow'}, style=edge_style),

        # Next pyramid levels (nested skip connections)
        StreamlitFlowEdge('4-13', '4', '13', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('9-13', '9', '13', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('10-13', '10', '13', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('5-14', '5', '14', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('10-14', '10', '14', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('11-14', '11', '14', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('6-15', '6', '15', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('11-15', '11', '15', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('12-15', '12', '15', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('4-16', '4', '16', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('9-16', '9', '16', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('13-16', '13', '16', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('14-16', '14', '16', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('5-17', '5', '17', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('10-17', '10', '17', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('14-17', '14', '17', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('15-17', '15', '17', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('4-18', '4', '18', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('9-18', '9', '18', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('13-18', '13', '18', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('16-18', '16', '18', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('17-18', '17', '18', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),

        # Embedding fusion for all decoder nodes
        StreamlitFlowEdge('2-13', '2', '13', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('3-13', '3', '13', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('2-14', '2', '14', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('3-14', '3', '14', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('2-15', '2', '15', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('3-15', '3', '15', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('2-16', '2', '16', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('3-16', '3', '16', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('2-17', '2', '17', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('3-17', '3', '17', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('2-18', '2', '18', animated=True, marker_end={'type': 'arrow'}, style=edge_style),
        StreamlitFlowEdge('3-18', '3', '18', animated=True, marker_end={'type': 'arrow'}, style=edge_style),

        # Final output
        StreamlitFlowEdge('18-19', '18', '19', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
        StreamlitFlowEdge('19-20', '19', '20', animated=True, marker_end={'type': 'arrow'}, style=black_edge_style),
    ]

    if 'static_flow_state_unetpp' not in st.session_state:
        st.session_state.static_flow_state_unetpp = StreamlitFlowState(nodes, edges)

    streamlit_flow('static_flow_unetpp',
                   st.session_state.static_flow_state_unetpp,
                   fit_view=True,
                   show_minimap=False,
                   show_controls=False,
                   pan_on_drag=True,
                   allow_zoom=True)


def instantiate_model_diagram(model_type:str, metadata_length:int=4):
    if model_type=='unet':
        instantiate_unet_diagram(metadata_length=metadata_length)
    elif model_type=='unet++':
        instantiate_unetpp_diagram(metadata_length=metadata_length)
