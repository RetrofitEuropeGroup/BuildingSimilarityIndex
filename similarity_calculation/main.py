import geopandas as gpd
from sklearn.metrics.pairwise import euclidean_distances

class similarity():
    def __init__(self, gpkg_path: str, columns: list):
        self.gpkg_path = gpkg_path
        self.gpf = gpd.read_file(self.gpkg_path)
        self.normalized_gpf = None
        self.columns = columns

    def normalize(self):
        """normalize the columns in the geopandas dataframe so that every feature has the same weight in the distance calculation"""
        self.normalized_gpf = self.gpf.copy()
        for column in self.columns:
            self.normalized_gpf[column] = (self.gpf[column] - self.gpf[column].mean()) / self.gpf[column].std()
    
    def calculate_distance(self, id1, id2):
        if self.normalized_gpf is None:
            self.normalize()

        if not id1.startswith("NL.IMBAG.Pand."):
            id1 = f"NL.IMBAG.Pand.{id1}-0"
        if not id2.startswith("NL.IMBAG.Pand."):
            id2 = f"NL.IMBAG.Pand.{id2}-0"
        
        obj1 = self.normalized_gpf[self.normalized_gpf["id"] == id1][self.columns]
        obj2 = self.normalized_gpf[self.normalized_gpf["id"] == id2][self.columns]

        dist = euclidean_distances(obj1, obj2)
        return dist[0][0]


if __name__ == '__main__':
    path = "collection/output/merged.gpkg"

    sim = similarity(path, ["roughness_index_3d", "actual_volume"])

    id1 = "NL.IMBAG.Pand.0327100000255061-0"
    id2 = "NL.IMBAG.Pand.0327100000264673-0"
     
    dist = sim.calculate_distance(id1, id2)