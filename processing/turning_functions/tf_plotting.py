import matplotlib.pyplot as plt
import os
import numpy as np


def plot_tf(tf):
    plt.stairs(tf[1], np.concatenate([[0.0], tf[0]]))


def plot_pollist(fs, titles=[]):
    if len(fs) > 10:
        tens = int(np.ceil(len(fs)/10))
        fig, axs = plt.subplots(tens, 10)
        plt.subplots_adjust(hspace=0.4)
        fig.set_figheight(3*tens)
        fig.set_figwidth(20)
        for i in range(len(fs)):
            x = int(i%10)
            y = int(np.floor(i/10))
            xmin = np.amin(fs[i].exterior.xy[0])
            xmax = np.amax(fs[i].exterior.xy[0])
            ymin = np.amin(fs[i].exterior.xy[1])
            ymax = np.amax(fs[i].exterior.xy[1])
            maxlen = max(xmax-xmin, ymax-ymin)
            axs[y, x].set_xlim(xmin-1, xmin+maxlen+1)
            axs[y, x].set_ylim(ymin-1, ymin+maxlen+1)
            #axs[x, y].set_xlim(-0.1, np.maximum(l1, l2)+0.6)
            #axs[x, y].set_ylim(-0.1, np.maximum(l1, l2)+0.6)
            axs[y, x].plot(*fs[i].exterior.xy)
            xx, yy = fs[i].exterior.xy
            axs[y, x].scatter(xx, yy, s=4)
            if titles != []:
                axs[y, x].set_title(titles[i])
    elif len(fs) > 1:
        fig, axs = plt.subplots(1, 10)
        plt.subplots_adjust(hspace=0.4)
        fig.set_figheight(2)
        fig.set_figwidth(20)
        for i in range(len(fs)):
            xmin = np.amin(fs[i].exterior.xy[0])
            xmax = np.amax(fs[i].exterior.xy[0])
            ymin = np.amin(fs[i].exterior.xy[1])
            ymax = np.amax(fs[i].exterior.xy[1])
            maxlen = max(xmax-xmin, ymax-ymin)
            axs[i].set_xlim(xmin-1, xmin+maxlen+1)
            axs[i].set_ylim(ymin-1, ymin+maxlen+1)
            #axs[x, y].set_xlim(-0.1, np.maximum(l1, l2)+0.6)
            #axs[x, y].set_ylim(-0.1, np.maximum(l1, l2)+0.6)
            axs[i].plot(*fs[i].exterior.xy)
            xx, yy = fs[i].exterior.xy
            axs[i].scatter(xx, yy, s=4)
            if titles != []:
                axs[i].set_title(titles[i])
    else:
        xmin = np.amin(fs[0].exterior.xy[0])
        xmax = np.amax(fs[0].exterior.xy[0])
        ymin = np.amin(fs[0].exterior.xy[1])
        ymax = np.amax(fs[0].exterior.xy[1])
        maxlen = max(xmax-xmin, ymax-ymin)
        plt.xlim(xmin-1, xmin+maxlen+1)
        plt.ylim(ymin-1, ymin+maxlen+1)
        plt.plot(*fs[0].exterior.xy)
        xx, yy = fs[0].exterior.xy
        plt.scatter(xx, yy, s=4)
        if titles != []:
            plt.title(titles[0])



## Plot all instances of single cluster
def get_cluster_indices(clusternr, clustering):
    return [i for i in range(len(clustering)) if clustering[i] == clusternr]

def save_clusters(clustering, destination="new_features/25_clusters/", pols=polygons):
    if not(os.path.exists(destination)):
        os.makedirs(destination)
    for i in range(len(list(set(clustering)))):
        plot_pollist([pols[i] for i in get_cluster_indices(clusternr=i, clustering=clustering)])
        plt.savefig(destination+"cluster_{0}.pdf".format(i), format='pdf')
        plt.close()


def plot_closest(ref_index, dataframe, pols=polygons, metric='l2', n_neighbors=20, filename="disance_from_polygon_", save=False, path="./"):
    ln = len(dataframe)
    distances = [dist(ref_index, i, dataframe) for i in range(ref_index)]+[0.0]+[dist(ref_index, i, dataframe) for i in range(ref_index+1, ln)]
    asort = np.argsort(distances)
    neighbor_indices = asort[:n_neighbors]
    plot_pollist([pols[i] for i in neighbor_indices], titles=["pol {0}: ".format(i)+str(np.round(distances[i], 3)) for i in neighbor_indices])
    if save:
        if not(os.path.exists(path)):
            os.makedirs(path)
        plt.savefig(path+filename+str(ref_index)+".pdf", format='pdf')
        subdat = dataframe.iloc[neighbor_indices]
        subdat.to_csv(path+"closest_to_pol_{0}.csv".format(ref_index), index=False)
    else:
        plt.show()
    plt.close()



def export_closest(ref_index, dataframe, pols=polygons, metric='l2', n_neighbors=20, filename="disance_from_polygon_", save=False, path="./", dist_columns = []):
    ln = len(dataframe)
    df_dist = dataframe.loc[:, dist_columns]
    distances = [dist(ref_index, i, df_dist) for i in range(ref_index)]+[0.0]+[dist(ref_index, i, df_dist) for i in range(ref_index+1, ln)]
    asort = np.argsort(distances)
    neighbor_indices = asort[:n_neighbors]
    # plot_pollist([pols[i] for i in neighbor_indices], titles=["pol {0}: ".format(i)+str(np.round(distances[i], 3)) for i in neighbor_indices])
    if save:
        if not(os.path.exists(path)):
            os.makedirs(path)
        #plt.savefig(path+filename+str(ref_index)+".pdf", format='pdf')
        subdat = dataframe.iloc[neighbor_indices]
        subdat.to_csv(path+"closest_to_pol_{0}.csv".format(ref_index), index=False)
    else:
        plt.show()
    plt.close()



