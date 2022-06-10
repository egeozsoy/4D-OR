import networkx as nx
from pyvis.network import Network
from matplotlib import pyplot as plt


def visualize_scene_graph(relations, save_path, use_pyvis=True):
    objs = set()
    for rel in relations:
        objs.add(rel[0])
        objs.add(rel[1])
    G = nx.DiGraph()

    for obj in objs:
        G.add_node(obj)

    edge_labels = {}
    for rel in relations:
        G.add_edge(rel[0], rel[1], label=rel[2])
        edge_labels[(rel[0], rel[1])] = rel[2]

    if use_pyvis:
        net = Network("900px", "900px", directed=True)
        net.from_nx(G)
        net.set_edge_smooth('dynamic')
        # net.show_buttons()
        net.set_options("""var options = {
        "layout":{"randomSeed":1},
      "nodes": {
        "borderWidth": 2,
        "color": {
          "border": "rgba(23,101,233,1)"
        },
        "font": {
          "color": "rgba(0,0,0,1)",
          "size": 15,
          "strokeWidth": 1
        },
        "scaling": {
          "min": 34,
          "max": 97
        },
        "shapeProperties": {
          "borderRadius": 5
        },
        "size": 71
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "dashes": true,
        "font": {
          "color": "rgba(255,15,20,1)",
          "size": 12
        },
        "smooth": {
          "roundness": 0.6
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "springLength": 100
        },
        "minVelocity": 0.22,
        "solver": "forceAtlas2Based",
        "timestep": 0.57
      }
    }""")
        net.show(save_path)
    else:
        pos = nx.spring_layout(G, k=10000, seed=1)
        nx.draw(G, pos, node_size=7500, node_color='#8ccdff')
        nx.draw_networkx_labels(G, pos, font_size=18)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=18)

        fig = plt.gcf()
        fig.set_size_inches(16, 10)
        plt.axis('off')
        axis = plt.gca()
        axis.set_xlim([1.25 * x for x in axis.get_xlim()])
        axis.set_ylim([1.25 * y for y in axis.get_ylim()])
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        plt.clf()
        plt.cla()
