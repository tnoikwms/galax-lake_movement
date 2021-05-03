from copy import deepcopy
from numpy import array
from numpy.linalg import norm
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class Node:
# A node represents a body if it is an endnote (i.e. if node.child is None)
# or an abstract node of the quad-tree if it has child.

    def __init__(self, m, x, y, z):
    # The initializer creates a child-less node (an actual body).
        self.m = m
        # Instead of storing the position of a node, we store the mass times
        # position, m_pos. This makes it easier to update the center-of-mass.
        self.m_pos = m * array([x, y, z])
        self.momentum = array([0., 0.,0.])
        self.child = None

    def into_next_quadrant(self):
    # Places node into next-level quadrant and returns the quadrant number.
        self.s = 0.5 * self.s   # s: side-length of current quadrant.
        #0,1,2,3で象限をかえす．(0,1,2,3)=(SE,NW,SE,NE)
        return self._subdivide(1) + 2*self._subdivide(0)+4*self._subdivide(2)

    def pos(self):
    # Physical position of node, independent of currently active quadrant.
        return self.m_pos / self.m

    def reset_to_0th_quadrant(self):
    # Re-positions the node to the level-0 quadrant (full domain).
        # Side-length of the level-0 quadrant is 1.
        self.s = 1.0
        # Relative position inside the quadrant is equal to physical position.
        self.relpos = self.pos().copy()

    def dist(self, other):
    # Distance between present node and another node.
        return norm(other.pos() - self.pos())

    def force_on(self, other):
    # Force which the present node is exerting on a given body.
        # To avoid numerical instabilities, introduce a short-distance cutoff.
        cutoff_dist = 0.002
        d = self.dist(other)
        if d < cutoff_dist:
            return array([0., 0., 0.])
        else:
            # Gravitational force goes like 1/r**2.
            #カットオフ以下のときだけ力を返す．
            return (self.pos() - other.pos()) * (self.m*other.m / d**3)

    def _subdivide(self, i):
    # Places node into next-level quadrant along direction i and recomputes
    # the relative position relpos of the node inside this quadrant.
        self.relpos[i] *= 2.0
        if self.relpos[i] < 1.0:
            quadrant = 0
        else:
            quadrant = 1
            #ここの-1はrelposを1以内に収めるためだけ，relposは実際の位置とはことなる
            #逐次relposはかわるはず．．．？じゃあ1以下にする必要なくない？これを消して結果がかわるか
            self.relpos[i] -= 1.0
        return quadrant


def add(body, node):
# Barnes-Hut algorithm: Creation of the quad-tree. This function adds
# a new body into a quad-tree node. Returns an updated version of the node.
    # 1. If node n does not contain a body, put the new body b here.
    #ここでのbodyはbodiesの配列要素，nodeを集めたもの
    new_node = body if node is None else None
    # To limit the recursion depth, set a lower limit for the size of quadrant.
    smallest_quadrant = 1.e-4
    if node is not None and node.s > smallest_quadrant:
        # 3. If node n is an external node, then the new body b is in conflict
        #    with a body already present in this region. ...
        if node.child is None:
            #idを変更してコピーされたnew_nodeをつくる
            #この段階では．new_nodeにはNoneが入る
            new_node = deepcopy(node)
        #    ... Subdivide the region further by creating four children
        #childrenには0,1,2,3が振られる，けどquadrantとは違います，
            new_node.child = [None for i in range(8)]
        #    ... And to start with, insert the already present body recursively
        #        into the appropriate quadrant.
        #ここでquadranを取得して，childrenの番号としている．
        #ただ，nodeと同じものが入るので，更新はされない，4つに増えるだけ
            quadrant = node.into_next_quadrant()
            new_node.child[quadrant] = node
        # 2. If node n is an internal node, we don't to modify its child.
        else:
            new_node = node

        # 2. and 3. If node n is or has become an internal node ...
        #           ... update its mass and "center-of-mass times mass".
        #ここで，center of massを計算する．総和をとっている．
        #でもnotNoneならNoneにしちゃってない？
        #いや上のelseでnew_nodeにnodeを入れてる
        new_node.m += body.m
        new_node.m_pos += body.m_pos
        # ... and recursively add the new body into the appropriate quadrant.
        quadrant = body.into_next_quadrant()
        #ここでやっと更新
        new_node.child[quadrant] = add(body, new_node.child[quadrant])
    return new_node


#1粒子にかかる力を計算
def force_on(body, node, theta):
# Barnes-Hut algorithm: usage of the quad-tree. This function computes
# the net force on a body exerted by all bodies in node "node".
# Note how the code is shorter and more expressive than the human-language
# description of the algorithm.
    # 1. If the current node is an external node, 
    #    calculate the force exerted by the current node on b.
    if node.child is None:
        return node.force_on(body)

    # 2. Otherwise, calculate the ratio s/d. If s/d < θ, treat this internal
    #    node as a single body, and calculate the force it exerts on body b.
    if node.s < node.dist(body) * theta:
        return node.force_on(body)

    # 3. Otherwise, run the procedure recursively on each child.
    return sum(force_on(body, c, theta) for c in node.child if c is not None)


#ここでGを書けるのね，なんでだろ
def verlet(bodies, root, theta, G, dt):
# Execute a time iteration according to the Verlet algorithm.
    for body in bodies:
        force = G * force_on(body, root, theta)
        body.momentum += dt * force
        body.m_pos += dt * body.momentum 

"""
#version errorがでるので直してね．
#ここがvisualizationのパート
def plot_bodies(bodies, i):
# Write an image representing the current position of the bodies.
# To create a movie with avconv or ffmpeg use the following command:
# ffmpeg -r 15 -i bodies_%06d.png -q:v 0 bodies.avi
#b.pos()[0] for b in bodies

"""

fig = plt.figure(figsize=(8,8),facecolor="w")
ax = fig.add_subplot(111,facecolor="w",projection="3d")
ax.set_xlim([0., 10.0])
ax.set_ylim([0., 10.0])
ax.set_zlim([0., 10.0])
ims=[]
ax.set_xticks([0.0,5.0,10.0])
ax.set_yticks([0.0,5.0,10.0])
ax.set_zticks([0.0,5.0,10.0])
######### MAIN PROGRAM ########################################################

# Theta-criterion of the Barnes-Hut algorithm.
theta = 0.5
# Mass of a body.
mass = 3.0
# Initially, the bodies are distributed inside a circle of radius ini_radius.
ini_radius = 0.5
# Initial maximum velocity of the bodies.
inivel = 0.1
# The "gravitational constant" is chosen so as to get a pleasant output.
G = 4.e-6
# Discrete time step.
dt = 1.e-3
# Number of bodies (the actual number is smaller, because all bodies
# outside the initial radius are removed).
numbodies = 300
# Number of time-iterations executed by the program.
max_iter = 20000
# Frequency at which PNG images are written.
img_iter = 20

# The pseudo-random number generator is initialized at a deterministic # value, for proper validation of the output for the exercise series.  random.seed(1)
# x- and y-pos are initialized to a square with side-length 2*ini_radius.
random.seed(1)
posx = random.random(numbodies) *2.*ini_radius + 5.0- ini_radius
posy = random.random(numbodies) *2.*ini_radius + 5.0- ini_radius
posz = random.random(numbodies) *0.01+5.0
# We only keep the bodies inside a circle of radius ini_radius.
bodies = [ Node(mass, px, py,pz) for (px,py,pz) in zip(posx, posy, posz) \
          if (px-5.0)**2. +(py-5.0)**2. < ini_radius**2.]

#input("Press the <ENTER> key to continue...")
#print ("here")
# Initially, the bodies have a radial velocity of an amplitude proportional to
# the distance from the center. This induces a rotational motion creating a
# "galaxy-like" impression.
for body in bodies:
    r = body.pos() - array([1.0,1.0,body.pos()[2]])
    body.momentum = array([-r[1], r[0] ,0.]) * mass*inivel*norm(r)/ini_radius

# Principal loop over time iterations.
for i in range(max_iter):
    # The quad-tree is recomputed at each iteration.
    root = None
    for body in bodies:
        body.reset_to_0th_quadrant()
        root = add(body, root)
    # Computation of forces, and advancment of bodies.
    verlet(bodies, root, theta, G, dt)

    # Output
    if i%img_iter==0:
        im = ax.scatter([b.pos()[0] for b in bodies], [b.pos()[1] for b in bodies],[b.pos()[2] for b in bodies],s=10,marker = ".",c="navy",alpha=0.3)
        ims.append([im])
ani = animation.ArtistAnimation(fig,ims,interval=50)

ani.save('galaxy_like_movement.mp4', writer="ffmpeg",dpi=150)
