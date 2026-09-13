import graphviz
import re

def visualize_computational_graph(raw_output):
    # Initialize the graph
    dot = graphviz.Digraph(comment='Autograd Graph', graph_attr={'rankdir': 'BT'}) 

    # Updated Regex to capture 'Is Updatable'
    pattern = r"Value (0x[0-9a-fA-F]+)\s+Data: ([\d\.-]+)\s+Grad: ([\d\.-]+)\s+isLeaf: (\d)\s+Child1 Pointer: (0x[0-9a-fA-F]+)\s+Child2 Pointer: (0x[0-9a-fA-F]+)\s+Operation:\s*(.)?\s+Is Updatable: (\d)"
    
    matches = re.findall(pattern, raw_output)
    seen_nodes = set()

    for addr, data, grad, is_leaf, c1, c2, op, is_updatable in matches:
        if addr in seen_nodes:
            continue
        seen_nodes.add(addr)

        # Labels
        op_label = op.strip() if op.strip() else "Leaf"
        # Using HTML-like labels for cleaner bolding
        node_label = f"{{ ADDR: {addr} | OP: {op_label} | DATA: {data} | GRAD: {grad} | UPDATABLE: {'YES' if is_updatable == '1' else 'NO'} }}"
        
        # Logic: 
        # is_leaf=1 -> Green-ish
        # is_leaf=0 -> Yellow-ish
        fill_color = '#e1f5fe' if is_leaf == '1' else '#fff9c4'
        
        # Logic:
        # Updatable -> Thick dashed red border
        # Constant -> Solid black border
        pen_width = '3' if is_updatable == '1' else '1'
        border_style = 'filled' if is_updatable == '1' else 'filled'
        border_color = '#d32f2f' if is_updatable == '1' else '#000000'

        dot.node(addr, 
                 label=node_label, 
                 shape='record', 
                 style=border_style, 
                 fillcolor=fill_color, 
                 color=border_color,
                 penwidth=pen_width)

        # Create edges
        if c1 != '0x0' and c1 != '(nil)':
            dot.edge(c1, addr, label="L")
        if c2 != '0x0' and c2 != '(nil)':
            dot.edge(c2, addr, label="R")

    return dot

# Paste your raw string here
data_input = """
Value 0x758c00e40
Data: 45.670078
Grad: 1.000000
isLeaf: 0
Child1 Pointer: 0x1036972b0
Child2 Pointer: 0x1036972b0
Operation: *
Is Updatable: 0

Value 0x1036972b0
Data: -6.757964
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x103697250
Child2 Pointer: 0x103697280
Operation: +
Is Updatable: 0

Value 0x103697250
Data: 1.242036
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x103697220
Child2 Pointer: 0x758c00e10
Operation: +
Is Updatable: 0

Value 0x103697220
Data: 0.431496
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x1036971f0
Child2 Pointer: 0x758c00de0
Operation: +
Is Updatable: 0

Value 0x1036971f0
Data: 0.096800
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x1036971a0
Child2 Pointer: 0x758c00db0
Operation: +
Is Updatable: 0

Value 0x1036971a0
Data: 0.000000
Grad: -13.515928
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00db0
Data: 0.096800
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x758c00d20
Child2 Pointer: 0x103696ec0
Operation: *
Is Updatable: 0

Value 0x758c00d20
Data: 0.200000
Grad: -6.541709
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696ec0
Data: 0.484000
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x103696e90
Child2 Pointer: 0x758c00ab0
Operation: +
Is Updatable: 0

Value 0x103696e90
Data: 0.471000
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x103696e60
Child2 Pointer: 0x758c00a80
Operation: +
Is Updatable: 0

Value 0x103696e60
Data: 0.307200
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x103696e10
Child2 Pointer: 0x758c00a50
Operation: +
Is Updatable: 0

Value 0x103696e10
Data: 0.000000
Grad: -2.703186
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00a50
Data: 0.307200
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x758c009c0
Child2 Pointer: 0x1036965a0
Operation: *
Is Updatable: 0

Value 0x758c009c0
Data: 0.480000
Grad: -1.730039
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x1036965a0
Data: 0.640000
Grad: -1.297529
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00a80
Data: 0.163800
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x758c009f0
Child2 Pointer: 0x1036965d0
Operation: *
Is Updatable: 0

Value 0x758c009f0
Data: 0.210000
Grad: -2.108485
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x1036965d0
Data: 0.780000
Grad: -0.567669
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00ab0
Data: 0.013000
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x758c00a20
Child2 Pointer: 0x103696600
Operation: *
Is Updatable: 0

Value 0x758c00a20
Data: 0.020000
Grad: -1.757071
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696600
Data: 0.650000
Grad: -0.054064
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00de0
Data: 0.334696
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x758c00d50
Child2 Pointer: 0x103696fe0
Operation: *
Is Updatable: 0

Value 0x758c00d50
Data: 0.340000
Grad: -13.305080
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696fe0
Data: 0.984400
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x103696fb0
Child2 Pointer: 0x758c00bd0
Operation: +
Is Updatable: 0

Value 0x103696fb0
Data: 0.878800
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x103696f80
Child2 Pointer: 0x758c00ba0
Operation: +
Is Updatable: 0

Value 0x103696f80
Data: 0.326800
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x103696f30
Child2 Pointer: 0x758c00b70
Operation: +
Is Updatable: 0

Value 0x103696f30
Data: 0.000000
Grad: -4.595416
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00b70
Data: 0.326800
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x758c00ae0
Child2 Pointer: 0x103696630
Operation: *
Is Updatable: 0

Value 0x758c00ae0
Data: 0.430000
Grad: -3.492516
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696630
Data: 0.760000
Grad: -1.976029
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00ba0
Data: 0.552000
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x758c00b10
Child2 Pointer: 0x103696660
Operation: *
Is Updatable: 0

Value 0x758c00b10
Data: 0.920000
Grad: -2.757249
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696660
Data: 0.600000
Grad: -4.227782
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00bd0
Data: 0.105600
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x758c00b40
Child2 Pointer: 0x103696690
Operation: *
Is Updatable: 0

Value 0x758c00b40
Data: 0.160000
Grad: -3.032974
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696690
Data: 0.660000
Grad: -0.735267
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00e10
Data: 0.810540
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x758c00d80
Child2 Pointer: 0x103697130
Operation: *
Is Updatable: 0

Value 0x758c00d80
Data: 0.900000
Grad: -12.172446
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103697130
Data: 0.900600
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x103697100
Child2 Pointer: 0x758c00cf0
Operation: +
Is Updatable: 0

Value 0x103697100
Data: 0.495600
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x1036970d0
Child2 Pointer: 0x758c00cc0
Operation: +
Is Updatable: 0

Value 0x1036970d0
Data: 0.183600
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x103697080
Child2 Pointer: 0x758c00c90
Operation: +
Is Updatable: 0

Value 0x103697080
Data: 0.000000
Grad: -12.164335
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00c90
Data: 0.183600
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x758c00c00
Child2 Pointer: 0x1036966c0
Operation: *
Is Updatable: 0

Value 0x758c00c00
Data: 0.340000
Grad: -6.568741
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x1036966c0
Data: 0.540000
Grad: -4.135874
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00cc0
Data: 0.312000
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x758c00c30
Child2 Pointer: 0x1036966f0
Operation: *
Is Updatable: 0

Value 0x758c00c30
Data: 0.520000
Grad: -7.298602
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x1036966f0
Data: 0.600000
Grad: -6.325454
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00cf0
Data: 0.405000
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x758c00c60
Child2 Pointer: 0x103696720
Operation: *
Is Updatable: 0

Value 0x758c00c60
Data: 0.540000
Grad: -9.123251
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696720
Data: 0.750000
Grad: -6.568741
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103697280
Data: -8.000000
Grad: -13.515928
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 0

Value 0x1036972b0
Data: -6.757964
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x103697250
Child2 Pointer: 0x103697280
Operation: +
Is Updatable: 0

Value 0x103697250
Data: 1.242036
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x103697220
Child2 Pointer: 0x758c00e10
Operation: +
Is Updatable: 0

Value 0x103697220
Data: 0.431496
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x1036971f0
Child2 Pointer: 0x758c00de0
Operation: +
Is Updatable: 0

Value 0x1036971f0
Data: 0.096800
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x1036971a0
Child2 Pointer: 0x758c00db0
Operation: +
Is Updatable: 0

Value 0x1036971a0
Data: 0.000000
Grad: -13.515928
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00db0
Data: 0.096800
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x758c00d20
Child2 Pointer: 0x103696ec0
Operation: *
Is Updatable: 0

Value 0x758c00d20
Data: 0.200000
Grad: -6.541709
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696ec0
Data: 0.484000
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x103696e90
Child2 Pointer: 0x758c00ab0
Operation: +
Is Updatable: 0

Value 0x103696e90
Data: 0.471000
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x103696e60
Child2 Pointer: 0x758c00a80
Operation: +
Is Updatable: 0

Value 0x103696e60
Data: 0.307200
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x103696e10
Child2 Pointer: 0x758c00a50
Operation: +
Is Updatable: 0

Value 0x103696e10
Data: 0.000000
Grad: -2.703186
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00a50
Data: 0.307200
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x758c009c0
Child2 Pointer: 0x1036965a0
Operation: *
Is Updatable: 0

Value 0x758c009c0
Data: 0.480000
Grad: -1.730039
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x1036965a0
Data: 0.640000
Grad: -1.297529
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00a80
Data: 0.163800
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x758c009f0
Child2 Pointer: 0x1036965d0
Operation: *
Is Updatable: 0

Value 0x758c009f0
Data: 0.210000
Grad: -2.108485
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x1036965d0
Data: 0.780000
Grad: -0.567669
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00ab0
Data: 0.013000
Grad: -2.703186
isLeaf: 0
Child1 Pointer: 0x758c00a20
Child2 Pointer: 0x103696600
Operation: *
Is Updatable: 0

Value 0x758c00a20
Data: 0.020000
Grad: -1.757071
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696600
Data: 0.650000
Grad: -0.054064
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00de0
Data: 0.334696
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x758c00d50
Child2 Pointer: 0x103696fe0
Operation: *
Is Updatable: 0

Value 0x758c00d50
Data: 0.340000
Grad: -13.305080
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696fe0
Data: 0.984400
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x103696fb0
Child2 Pointer: 0x758c00bd0
Operation: +
Is Updatable: 0

Value 0x103696fb0
Data: 0.878800
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x103696f80
Child2 Pointer: 0x758c00ba0
Operation: +
Is Updatable: 0

Value 0x103696f80
Data: 0.326800
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x103696f30
Child2 Pointer: 0x758c00b70
Operation: +
Is Updatable: 0

Value 0x103696f30
Data: 0.000000
Grad: -4.595416
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00b70
Data: 0.326800
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x758c00ae0
Child2 Pointer: 0x103696630
Operation: *
Is Updatable: 0

Value 0x758c00ae0
Data: 0.430000
Grad: -3.492516
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696630
Data: 0.760000
Grad: -1.976029
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00ba0
Data: 0.552000
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x758c00b10
Child2 Pointer: 0x103696660
Operation: *
Is Updatable: 0

Value 0x758c00b10
Data: 0.920000
Grad: -2.757249
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696660
Data: 0.600000
Grad: -4.227782
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00bd0
Data: 0.105600
Grad: -4.595416
isLeaf: 0
Child1 Pointer: 0x758c00b40
Child2 Pointer: 0x103696690
Operation: *
Is Updatable: 0

Value 0x758c00b40
Data: 0.160000
Grad: -3.032974
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696690
Data: 0.660000
Grad: -0.735267
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00e10
Data: 0.810540
Grad: -13.515928
isLeaf: 0
Child1 Pointer: 0x758c00d80
Child2 Pointer: 0x103697130
Operation: *
Is Updatable: 0

Value 0x758c00d80
Data: 0.900000
Grad: -12.172446
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103697130
Data: 0.900600
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x103697100
Child2 Pointer: 0x758c00cf0
Operation: +
Is Updatable: 0

Value 0x103697100
Data: 0.495600
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x1036970d0
Child2 Pointer: 0x758c00cc0
Operation: +
Is Updatable: 0

Value 0x1036970d0
Data: 0.183600
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x103697080
Child2 Pointer: 0x758c00c90
Operation: +
Is Updatable: 0

Value 0x103697080
Data: 0.000000
Grad: -12.164335
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00c90
Data: 0.183600
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x758c00c00
Child2 Pointer: 0x1036966c0
Operation: *
Is Updatable: 0

Value 0x758c00c00
Data: 0.340000
Grad: -6.568741
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x1036966c0
Data: 0.540000
Grad: -4.135874
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00cc0
Data: 0.312000
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x758c00c30
Child2 Pointer: 0x1036966f0
Operation: *
Is Updatable: 0

Value 0x758c00c30
Data: 0.520000
Grad: -7.298602
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x1036966f0
Data: 0.600000
Grad: -6.325454
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x758c00cf0
Data: 0.405000
Grad: -12.164335
isLeaf: 0
Child1 Pointer: 0x758c00c60
Child2 Pointer: 0x103696720
Operation: *
Is Updatable: 0

Value 0x758c00c60
Data: 0.540000
Grad: -9.123251
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103696720
Data: 0.750000
Grad: -6.568741
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 1

Value 0x103697280
Data: -8.000000
Grad: -13.515928
isLeaf: 1
Child1 Pointer: 0x0
Child2 Pointer: 0x0
Operation: l
Is Updatable: 0"""
graph = visualize_computational_graph(data_input)
graph.render('my_neural_net', format='png', view=True)