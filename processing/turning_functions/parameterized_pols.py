import shapely


## lengte zijde 1, lengte zijde 2
def rechthoekig(l1, l2):
    return shapely.Polygon(((0.0, 0.0), (l1, 0.0), (0.0, l2)))

## Breedte, hoogte
def gelijkbenig(l1, l2):
    return shapely.Polygon(((0.0, 0.0), (l1, 0.0), (l1/2, l2)))




## breedte, hoogte
def rect(l1, l2):
    return shapely.Polygon(((0.0, 0.0), (l1, 0.0), (l1, l2), (0.0, l2)))

## Breedte, dikte1, hoogte, dikte2
def L(l1, l2, t1, t2):
    return shapely.Polygon(((0.0, 0.0), (l1, 0.0), (l1, t1), (t2, t1), (t2, l2), (0.0, l2)))

# width, height, thickness
def U(w, h, t):
    return shapely.Polygon(((0.0, 0.0), (w+t/2, 0.0), (w+t/2, h), (w-t/2, h), (w-t/2, t), (t, t), (t, h), (0, h)))

def T(w, h, t):
    return shapely.Polygon(((0.0, 0.0), (w, 0.0), (w, t), ((w+t)/2, t), (((w+t)/2), h), (((w-t)/2), h), (((w-t)/2), t), (0.0, t)))

## O

def O(b, h, d):
    return shapely.Polygon(((0.0, 0.0), (b+(d/2), 0.0), (b+(d/2), h+(d/2)), (0.0, h+(d/2)), (0.0, 2*d), (d, 2*d), (d, h-(d/2)), (b-(d/2), h-(d/2)), (b-(d/2), d), (0.0, d)))



def huis(b, h1, h2):
    return shapely.Polygon(((0.0, 0.0), (b, 0.0), (b, h1), ((b/2), h1+h2), (0.0, h1), (0.0, 0.0)))



def H(b, h, d):
    return shapely.Polygon(((0.0, 0.0), (d, 0.0), (d, (h/2)-(d/2)), (b-(d/2), (h/2)-(d/2)), (b-(d/2), 0.0), (b+(d/2), 0.0), (b+(d/2), h), (b-(d/2), h), (b-(d/2), (h/2)+(d/2)), (d, (h/2)+(d/2)), (d, h), (0.0, h)))


def F(h, b1, b2, d):
    return shapely.Polygon(((0.0, 0.0), (d, 0.0), (d, (h/2)-(d/2)), (b1, (h/2)-(d/2)), (b1, (h/2)+(d/2)), (d, (h/2)+(d/2)), (d, h-d), (b2, h-d), (b2, h), (0.0, h)))


def Z(b, h, d):
    return shapely.Polygon(((0.0, 0.0), (b, 0.0), (b, d), (d, d), (b, h-d), (b, h), (0.0, h), (0.0, h-d), (b-d, h-d), (0.0, d)))


def P(h, b, d):
    return shapely.Polygon(((0.0, 0.0), (d, 0.0), (d, h-d), (b-d, h-d), (b-d, (h/2)+(d/2)), (2*d, (h/2)+(d/2)), (2*d, (h/2)-(d/2)), (b, (h/2)-(d/2)), (b, h), (0.0, h)))

def regular(sides):
    point_list = [(np.cos(i*(2*np.pi/sides)), np.sin(i*(2*np.pi/sides))) for i in range(sides)]
    return shapely.Polygon(point_list)
