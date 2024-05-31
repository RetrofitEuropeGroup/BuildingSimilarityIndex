import tf

data_path = "."

from sklearn.preprocessing import MinMaxScaler
def process_to_features(filenames, geometry_features=feat_small, path="./", geometry_column='geometry', other_columns = ['bouwjaar', 'a_vb', 'a_vb_wf', 'a_p', 'c_area', 'maxz.max', 'h_dak_70p.max'], scaling=True, scaler=MinMaxScaler(), metric='l2', relative_feature_weight = False, categorical_columns = []):
    ## One file provided:
    def file_to_df(filenames, geometry_features=feat_small, path="./", geometry_column='geometry', other_columns = ['bouwjaar', 'a_vb', 'a_vb_wf', 'a_p', 'c_area', 'maxz.max', 'h_dak_70p.max'], scaling=True, scaler=MinMaxScaler(), metric='l2', relative_feature_weight = False):
        data = gpd.read_file(path+filenames)
        pols = list(data[geometry_column])
        if type(pols[0]) == shapely.geometry.MultiPolygon:
            pols = [list(i.geoms)[0] for i in pols]
        pols = [round_polygon(translate_pol(i)) for i in pols]
        new_df = pd.DataFrame(make_space(pols, features=[pol_to_vec(i) for i in geometry_features], metric='l2'))
        df_other = data[other_columns]
        df_c = data[categorical_columns]
        # df_c = 
        if scaling:
            df_other = scaler.fit_transform(df_other)
        if relative_feature_weight:
            df_other = np.multiply(df_other, np.sqrt(len(geometry_features)))
        df_other = pd.DataFrame(df_other)
        df = pd.concat([new_df.reset_index(drop=True), df_other.reset_index(drop=True)], axis=1)
        return df, pols
    if type(filenames) == type("hello"):
        total_df, pols = file_to_df(filenames, geometry_features=geometry_features, path=path, geometry_column=geometry_column, other_columns=other_columns, scaling=scaling, scaler=scaler, metric=metric, relative_feature_weight=relative_feature_weight)
    elif type(filenames) == type([1, 2, 3]) and filenames != []:
        total_df, pols = file_to_df(filenames[0], geometry_features=geometry_features, path=path, geometry_column=geometry_column, other_columns=other_columns, scaling=scaling, scaler=scaler, metric=metric, relative_feature_weight=relative_feature_weight)
        for fn in filenames[1:]:
            new_df, new_pols = file_to_df(fn, geometry_features=geometry_features, path=path, geometry_column=geometry_column, other_columns=other_columns, scaling=scaling, scaler=scaler, metric=metric, relative_feature_weight=relative_feature_weight)
            total_df = total_df.append(new_df, ignore_index=True)
            pols = pols + new_pols
    total_df.columns = list(range(len(total_df.columns)))
    return total_df, pols
