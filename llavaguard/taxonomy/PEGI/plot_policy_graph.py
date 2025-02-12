import networkx as nx
import matplotlib.pyplot as plt
# Create an undirected graph
G = nx.Graph()

# Add policy as the root node
G.add_node("Policy")

# Add nodes and edges from safety policy
for main_category, details in safety_policy.items():
    # create nodes for main category
    G.add_node(main_category)
    # create edges between policy and main category
    G.add_edge("Policy", main_category)
    
    for subcategory, elements in details['Sub-Categories'].items():
        # Create edges between main category and each subcategory
        i = 0
        while subcategory in G.nodes:
            subcategory = subcategory.split('_')[0] + str(i)
        G.add_node(subcategory)
        G.add_edge(main_category, subcategory)
        # Create edges between subcategory and each detail within it
        # for element in elements:
        #     G.add_edge(subcategory, element)

    for rating, subcategories in details['Rating'].items():
        # Create edges between main_category and each rating
        G.add_node(rating)
        # G.add_edge(main_category, rating)
        # Create edges between rating and subcategories associated with it
        for subcategory in subcategories:
            G.add_edge(rating, subcategory)

# Draw the graph
plt.figure(figsize=(30, 30))
main_categories = list(safety_policy.keys())
circular_pos = nx.circular_layout(main_categories)
pos = nx.spring_layout(G, seed=42, iterations=1000, k=.35, pos=circular_pos)

nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=8, font_weight='bold')
plt.title("Safety Policy Graph")
plt.savefig("safety_policy_graph.png", dpi=100)