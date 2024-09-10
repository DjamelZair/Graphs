""" Creating an app for the railway


Djamel Zair 11015934"""

# I will be using pandas to read the json files
import pandas as pd
import json
from paths import AdjListsWeightedGraph
from paths import reconstruct_path


# making shortcuts to the needed files
GRAPHS = "data\\treingraaf_graphs.json"
LINKS = "data\\treingraaf_links.json"
STATIONS = "data\\treingraaf_stations.json"

# This is still hardcoded for now
N_STATIONS = 421

# Now I will be seperating the two graphs from the links json  file.
with open(LINKS, "r") as f:
    data = json.load(f)

# Convert dictionary into a DataFrame
df = pd.DataFrame(data)

links = []
for item in data:
    for row in item["data"]:
        links.append(row)

df = pd.DataFrame(links)

# Separate rows by graph_id
graph1_df = df[df["graph_id"] == "1"]  # Tariefeenheden
graph2_df = df[df["graph_id"] == "2"]  # Kilometers

# Initialize the graph
graph_train = AdjListsWeightedGraph(N_STATIONS)

# Now I want all the stations in a dataframe so we can get their ID
with open(STATIONS, "r") as f:
    data_stations = json.load(f)

# Convert dictionary into a DataFrame
df_s = pd.DataFrame((data_stations[0]["data"]))
print(df_s.head())


def navigation(graph: int, destination: int, source: int):

    if graph == 1:
        for row in graph1_df.itertuples(index=False):
            graph_train.add_edge(
                int(row.begin_id) - 1, int(row.end_id) - 1, int(row.value)
            )
            print("Initializing navigation based on 'tariefeenheden'")
            cost = "tariefeenheden"

    elif graph == 2:
        for row in graph2_df.itertuples(index=False):
            graph_train.add_edge(
                int(row.begin_id) - 1, int(row.end_id) - 1, int(row.value)
            )
            print("Initializing navigation based on 'kilometers'")
            cost = "kilometers"

    # Getting shortest path to aLL other nodes from the source node
    pred, dist = graph_train.dijkstra(source=source)

    # From the list of all the costs I want the one of the destination node
    dist = dist[destination - 1]

    # Now calculating the specific path to the destination
    best_path = reconstruct_path(pred, destination)
    print(f"Your most optimal route based on {cost}")
    print(f"has {dist} {cost}")
    return best_path


def main():
    destination = str(input("Please provide place of destination : "))
    source = str(input("Please provide place of departure: "))
    graph_choice = int(
        input("For 'tariefeenheden' choose 1, for 'kilometers' choose 2: ")
    )

    # find row(s) where long_name matches user input
    matching_dest = df_s[df_s["long_name"].str.lower() == destination.lower()]
    matching_sour = df_s[df_s["long_name"].str.lower() == source.lower()]

    # Checking if the station exists
    if matching_dest.empty:
        print(f"No station found with name '{destination}' Try again?")
        if str(input("y/n")).lower() == "y":
            main()
    if matching_sour.empty:
        print(f"No station found with name '{source}' Try again")
        if str(input("y/n")).lower() == "y":
            main()

    # extract id values
    id_dest = int(matching_dest["id"])
    id_sour = int(matching_sour["id"])

    # Putting information in my navigation function
    navigation(graph_choice, id_dest, id_sour)


if __name__ == "__main__":
    main()
