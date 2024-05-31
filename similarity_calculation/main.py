import geopandas as gpd
from sklearn.metrics.pairwise import euclidean_distances

class similarity:
    def __init__(self, gpkg_path: str, column_weights: dict = None, columns: list = None):
        self.gpkg_path = gpkg_path
        self.gpf = gpd.read_file(self.gpkg_path)
        self.normalized_gpf = None

        self._validate_input(column_weights, columns)

        self.column_weights = column_weights
        self.columns = self._get_columns(column_weights, columns)
    
    ## helper functions for __init__
    def _validate_input(self, column_weights, columns):
        if column_weights is not None and not isinstance(column_weights, dict):
            raise ValueError("column_weights must be a dictionary")
        if columns is not None and not isinstance(columns, list):
            raise ValueError("columns must be a list")

    def _get_columns(self, column_weights, columns):
        if column_weights is not None:  # overwrite columns if column_weights is given
            return list(column_weights.keys())
        elif columns is None and column_weights is None:
            return self.gpf.columns.drop(["id", "geometry"])
        else:
            return columns
        
    def normalize(self):
        """normalize the columns in the geopandas dataframe so that every feature has the same weight in the distance calculation"""
        self.normalized_gpf = self.gpf.copy()
        for column in self.columns:
            self.normalized_gpf[column] = (self.gpf[column] - self.gpf[column].mean()) / self.gpf[column].std()
    
    def scale(self):
        pass


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

    sim = similarity(path)

    id1 = "NL.IMBAG.Pand.0327100000255061-0"
    id2 = "NL.IMBAG.Pand.0327100000264673-0"
     
    dist = sim.calculate_distance(id1, id2)