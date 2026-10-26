import json
import geopandas as gpd
import geojson
import math
import requests



def _face_area(face_coords: list, vertices: list, scale: list) -> float:
    n = len(face_coords)
    if n < 3:
        return 0.0

    # Shoelace formula
    area = 0.0
    for i in range(n):
        v1 = vertices[face_coords[i]]
        v2 = vertices[face_coords[(i + 1) % n]]
        area += v1[0] * v2[1] - v2[0] * v1[1]

    return abs(area) * 0.5 * (scale[0] * scale[1])


def _total_area(face_coords: list, vertices: list, scale: list) -> float:
    exterior_area = _face_area(face_coords[0], vertices, scale)
    interior_area = sum(_face_area(hole, vertices, scale) for hole in face_coords[1:]) # Holes
    return exterior_area - interior_area


def get_roof_attributes(
        bagid: str, 
        parent_cj: dict, 
        childs_cj: dict, 
        vertices: list,
        cj_scale: list, 
        verbose: bool = False
    ) -> dict:
    
    # Template by Claude
    roof_parts = []
    for child_id in parent_cj["children"]:
        for geom in childs_cj[child_id]["geometry"]:
            if geom["lod"] != "2.2":
                continue
            # if not geom["semantics"]:   # TODO: rm if unn
            #     continue

            surfaces = geom.get("semantics").get("surfaces")
            values = geom.get("semantics").get("values")

            # TODO: Add other geometry types
            if not (geom.get("type") == "Solid"):
                if verbose:
                    print(f"Geometry is not Solid. It's: {geom.get('type')}.")
                continue
            
            for i, boundary_group in enumerate(geom.get("boundaries")):
                bound_group_values = values[i]
                for j, boundary_points in enumerate(boundary_group):
                    boundary_value = bound_group_values[j]

                    boundary_sem = surfaces[boundary_value]
                    if not boundary_sem.get("type") == "RoofSurface":
                        continue

                    boundary_2d_area = _total_area(boundary_points, vertices, cj_scale)

                    # TODO: 3d area
                    # if use_3d_area and surf_sem.get("b3_hellingshoek") is not None:
                    #     slope_rad = math.radians(surf_sem["b3_hellingshoek"])
                    #     cos_val = math.cos(slope_rad)
                    #     area = area_2d / cos_val if cos_val > 1e-6 else area_2d
                    # else:
                    #     area = area_2d
                    area = boundary_2d_area

                    boundary_sem.update({"area": area})
                    roof_parts.append(boundary_sem)

    pand_max_height = max(part["b3_h_dak_max"] for part in roof_parts)
    
    # Main roof (mr) and other roof (other) parts
    mr_no_parts = 0
    mr_area = 0.0
    other_no_parts = 0
    other_area = 0.0

    for part in roof_parts:
        if (part["b3_h_dak_max"] > pand_max_height - 1.0) or (part["area"] > 50.0):
            mr_no_parts += 1
            mr_area += part["area"]
        else:
            other_no_parts += 1
            other_area += part["area"]

    part_ratio = mr_no_parts / (mr_no_parts + other_no_parts)
    area_ratio = mr_area / (mr_area + other_area)        

    if verbose:
        print(f'Building: {bagid}')
        print(f'Main roof: {mr_no_parts} parts with a total area of {mr_area:.2f} m^2')
        print(f'Other roof: {other_no_parts} parts with a total area of {other_area:.2f} m^2')
        print(f'Part ratio: {part_ratio:.2f}')
        print(f'Area ratio: {area_ratio:.2f}')
    
    return dict(
        main_roof_parts=int(mr_no_parts),
        main_roof_area=mr_area,
        other_roof_parts=int(other_no_parts),
        other_roof_area=other_area,
        part_ratio=part_ratio,
        area_ratio=area_ratio
    )
