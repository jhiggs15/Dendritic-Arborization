#%%
#@title Imports and Helper function definitions
import os
import io
import base64
import tqdm
import requests
import PIL.Image, PIL.ImageDraw
import numpy as np
import random
import matplotlib.pylab as pl
from IPython.display import Image, HTML, clear_output
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

os.environ['FFMPEG_BINARY'] = 'ffmpeg'

clear_output()

def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def im2url(a, fmt='jpeg'):
  encoded = imencode(a, fmt)
  base64_byte_string = base64.b64encode(encoded).decode('ascii')
  return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

def imshow(a, fmt='jpeg'):
  display(Image(data=imencode(a, fmt)))

def tile2d(a, w=None):
  a = np.asarray(a)
  if w is None:
    w = int(np.ceil(np.sqrt(len(a))))
  th, tw = a.shape[1:3]
  pad = (w-len(a))%w
  a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
  h = len(a)//w
  a = a.reshape([h, w]+list(a.shape[1:]))
  a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
  return a

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img

class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()

def load_image(url, max_size=256):
  r = requests.get(url)
  img = PIL.Image.open(io.BytesIO(r.content))
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)/255.0
  # premultiply RGB by Alpha
  img[..., :3] *= img[..., 3:]
  return img

def load_emoji(emoji):
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u%s.png'%code
  return load_image(url)


def to_rgba(x):
  return x[..., :4]

def to_alpha(x):
  return np.clip(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
  # assume rgb premultiplied by alpha
#   rgb, a = x[..., :3], to_alpha(x)
#   return 1.0-a+rgb
  return x[..., :3]

def make_seed(size, n=1):
  x = np.zeros([n, size, size, CHANNEL_N], np.float32)
  x[:, size//2, size//2, 3:] = 1.0
  return x

#%%
max_identifier = 0
neuron_count = 1
num_cells = 1
size = 256
world = np.zeros([size, size], np.uint64)

dct = {}
growth_cones = []

#%%
neuron_types = {
    1 : {
        'color' : [135,206,250, 255], #sky blue
        'color_stem' : [0, 0, 255, 255], #blue
        'color_gc' : [72,61,139, 255], #dar
        'weights' : [32,62,6]#[40, 45, 5] # move, emit, branch
    },
    2 : {
        'color' : [0,255,127, 255], #spring green
        'color_stem' : [50,205,50, 255], #limegreen 
        'color_gc' : [0,100,0, 255], #dark green
        'weights' : [32,62,6]#[60, 35, 5] # move, emit, branch
    },
    3 : {
        'color' : [127, 127, 255, 255], # light purple
        'color_stem' : [127, 63, 127, 255], # darker purple
        'color_gc' : [127, 63, 63, 255], # violet
        'weights' : [32,62,6]#[50, 35, 15] # move, emit, branch
    },
    4 : {
        'color' : [255,182,193, 255], #light pink
        'color_stem' : [255,20,147, 255], #deep pinks
        'color_gc' : [199,21,133, 255], #violet
        'weights' : [32,62,6]#[60, 30, 10] # move, emit, branch
    }
}

def spawn_growth_cone(w, parent_id, x, y, energy = 50, heading=(5, 5)):
    global max_identifier
    global world
    global dct
    max_identifier += 1
    growth_cones.append(max_identifier)
    dct[max_identifier] = {
        'itemtype': "growth cone",
        'pos' : (x, y),
        'parent_id' : parent_id,
        'energy' : energy,
        'color' : neuron_types[dct[parent_id]['neuron_type']]['color_gc'], 
        'split_cnt' : random.randint(2,6),
        'heading' : heading,
        'neuron_type' : dct[parent_id]['neuron_type']
    }
    w[x][y] = max_identifier    
    
# add expression 
def spawn_neuron(w, neuron_type, pos):
    global max_identifier
    global world
    global dct
    max_identifier += 1
    dct[max_identifier] ={
        'itemtype': "neuron",
        'neuron_type': neuron_type,
        'pos': pos,
        'energy': 200,
        'color' : neuron_types[neuron_type]['color']
    }
    w[pos[0]][pos[1]] = max_identifier
    heading = random.choice([(-1, -1), (-1, 0), (-1,1), (0, -1), (0, 1), (1, -1), (1, 0), (1,1)])
    spawn_growth_cone(w, max_identifier, pos[0], pos[1], energy = 60, heading = heading)

def spawn_stem(w, parent_id, x, y):
  global max_identifier
  global dct
  max_identifier += 1
  growth_cones.append(max_identifier)
  dct[max_identifier] = {
      'itemtype': "stalk",
      'pos' : (x,y),
      'color' : neuron_types[dct[parent_id]['neuron_type']]['color_stem'], 
      'parent_id' : parent_id,
      'neuron_type' : dct[parent_id]['neuron_type']
  }
  w[x][y] = max_identifier  

def spawn_dscam(w, parent_id, x, y):
  global max_identifier
  global dct
  max_identifier += 1
  dct[max_identifier] = {
      'itemtype': "dscam",
      'pos' : (x,y),
      'color' : dct[parent_id]['color'],
      'parent_id' : parent_id
  }
  w[x][y] = max_identifier 

#%%

# What interact will do: return a list of commands for each possible thing it can do
#   if its a growth cone:
#     add move command, based on heading 
#     Later:
#       evaluate its surroundings (maybe later when dscam is here)
#       evaluate move based on heading and dscam (but for now just heading)
#     
#   if anything else:
#     do nothing

def find_fin(cur_x, cur_y, head_x, head_y):
  sign_x = head_x < 0
  sign_y = head_y < 0

  if abs(head_x) > 0.5:
    cur_x -= 1 if sign_x else -1
    if sign_x:
      head_x = -1
    else:
     head_x = 1
  else:
    head_x = 0
  if abs(head_y) > 0.5:
    cur_y -= 1 if sign_y else -1
    if sign_x:
      head_y = -1
    else:
      head_y = 1
  else:
    head_y = 0
  return (cur_x, cur_y, head_x, head_y)


def move_cmd(focused_id):
    global dct
    head = dct[focused_id]['heading']
    head_x = head[0]
    head_y = head[1]
    cur_x = dct[focused_id]['pos'][0]
    cur_y = dct[focused_id]['pos'][1]

    res = find_fin(cur_x, cur_y, head_x, head_y)

    mv = {
        'cmd' : "move",
        'fid' : focused_id,
        'org_x' : cur_x,
        'org_y' : cur_y,
        'head_x' : head_x,
        'head_y' : head_y,
        'fin_x' : res[0],
        'fin_y' : res[1],
        'unit_x' : res[2],
        'unit_y' : res[3]
    }
    return mv

def emit_cmd(focused_id):
    global dct
    head = dct[focused_id]['heading']
    f_x = dct[focused_id]['pos'][0]
    f_y = dct[focused_id]['pos'][1]
    emit = {
        'cmd' : "emit",
        'fid' : focused_id,
        'org_x' : f_x,
        'org_y' : f_y,
        'head_x' : head[0],
        'head_y' : head[1]
    }
    return emit

def branch_cmd(focused_id):
    global dct
    head = dct[focused_id]['heading']
    gc_x = dct[focused_id]['pos'][0]
    gc_y = dct[focused_id]['pos'][1]
    # finds heading perpendicular
    new_head = (head[1]*-1, head[0])
    # fifty percent chance to check if both are negative to reduce sign
    if random.uniform(0,1) > .20:
      if new_head[0] < 0 and new_head[1] < 0:
        a = abs(new_head[0])
        b = abs(new_head[1])
        new_head = (a,b)
      else:
        a = new_head[0] * -1
        b = new_head[1] * -1
        new_head = (a,b)

    if new_head[0] < 0:
      gc_x = gc_x - 1
    elif new_head[0] > 0:
      gc_x = gc_x + 1
    else:
      print("0!")
    if new_head[1] < 0:
      gc_y = gc_y - 1
    elif new_head[1] > 0:
      gc_y = gc_y + 1
    else:
      print("0 y")
    brnch = {
        'cmd' : "branch",
        'fid' : focused_id,
        'gc_x' : gc_x,
        'gc_y' : gc_y,
        'head_x' : new_head[0],
        'head_y' : new_head[1]
    }
    return brnch


    # would normally calculate all possible moves
    # choose one randomly to return
    #   Randomness comes from energy and how much it has
    #   if not enough energy it will just stay stagnant
    #focus_x = dct[focused_id]['pos'][0]
    #focus_y = dct[focused_id]['pos'][1]
    # for i in [-1, 0, 1]:
    #     for j in [-1, 0, 1]:
    #         try:
    #             neigbor_id = world[focus_x + i][focus_y + j]
    #         except KeyError:
    #             pass

    #only action now is move
            
    # if dct[focused_id]['itemtype'] == "growth cone" :
    #     #if it is looking at itself
    #     if neighbor_id != 0:

def interact(world, focused_id):
    global max_identifier
    global dct

    neuron_type = dct[focused_id]['neuron_type']
    weights = neuron_types[neuron_type]['weights']
    opts = ["move", "emit", "branch"]
    
    action = random.choices(opts, weights)[0]

    if action == "branch":
      return branch_cmd(focused_id)
    elif action == "move":
      return move_cmd(focused_id)
    elif action == "emit":
      return emit_cmd(focused_id)

# can be modified 
def vision(u_x, u_y):
  clock = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
  org_ind = clock.index(((u_x, u_y)))
  vis = [clock[org_ind], clock[(org_ind + 1) % 8], clock[(org_ind + 2) % 8], clock[(org_ind - 1) % 8], clock[(org_ind - 2) % 8]]
    
  return vis


# makes all needed changes in the world
def execute_actions(cmds, new_world):
    global dct
    if len(cmds) == 0:
        return new_world
    # print(cmds)
    #action = random.choice(cmds)
    for action in cmds:
        cmd = action['cmd']
        fid = action['fid']
        if cmd == "move":
            #grabbing needed dictionary values
            fin_x = action['fin_x']
            fin_y = action['fin_y']
            head_x = action['head_x']
            head_y = action['head_y']
            fid = action['fid']
            cur_x = action['org_x']
            cur_y = action['org_y']
            mag = np.sqrt(head_x ** 2 + head_y ** 2)
            u_x = action['unit_x']
            u_y = action['unit_y']
            #checks to see if the cone should be removed 
            rmv_cone = False
            try:
                #spawns stem in place of growth cone
                if not new_world[fin_x][fin_y] == 0 and not (dct[new_world[fin_x][fin_y]]['itemtype'] == "dscam" and dct[new_world[fin_x][fin_y]]['parent_id'] == dct[fid]['parent_id']):
                    #remove cone it is headerd toward either a cone or a stem
                    rmv_cone = True
                if not rmv_cone:
                    # scan in the direction of the current header to make 
                    # adjustments based on dscam 
                    # vision is approx 180 degrees
                    #vis = vision(u_x, u_y)
                    # How far in front of the growth cone it can see
                    head_change = (0,0)
                    x = cur_x
                    y = cur_y
                    cur_vec = (x, y, 0, 0)
                    for j in [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:
                      for i in range(1,4):
                        cur_vec = find_fin(cur_vec[0], cur_vec[1], j[0], j[1])
                        if not (cur_vec[0] > 255 or cur_vec[1] > 255 or cur_vec[0] < 0 or cur_vec[1] < 0):
                          new_world[cur_vec[0]][cur_vec[1]]
                          ind = new_world[cur_vec[0]][cur_vec[1]]
                          if ind != 0 and dct[ind]['itemtype'] == "dscam" and not (dct[ind]['parent_id'] == dct[fid]['parent_id']):
                            head_change = (head_change[0] + cur_vec[2] * -1, head_change[1] + cur_vec[3] * -1)
                        else:
                          pass
                    if not head_change[0] == 0 and not head_change[1] == 0:
                      new_head_x = head_change[0] + head_x
                      new_head_y = head_change[1] + head_y
                      new_sign_x = new_head_x < 0
                      new_sign_y = new_head_y < 0
                      if not new_head_x == head_x * -1 and not new_head_y == head_y * -1:
                        new_fin = find_fin(cur_x, cur_y, new_head_x, new_head_y)
                        fin_x = new_fin[0]
                        fin_y = new_fin[1]
                        if new_sign_x:
                          head_x = head_x * -1
                        if new_sign_y:
                          head_y = head_y * -1
                      else:
                        new_world[cur_x][cur_y] = 0
                        fin_x = cur_x
                        fin_y = cur_y
                    
                    # randomize heading a little
                    heading_drift = 0.2
                    head_x = max(-1, min(head_x + (random.random()*2 - 1) * heading_drift, 1.0))
                    head_y = max(-1, min(head_y + (random.random()*2 - 1) * heading_drift, 1.0))
                    # updates its dictionary entry  
                    if not (cur_x == fin_x and cur_y == fin_y):
                      spawn_stem(new_world, dct[fid]['parent_id'], cur_x, cur_y)
                    dct[fid]['pos'] = (fin_x,fin_y)
                    dct[fid]['heading'] = (head_x, head_y)
                    new_world[fin_x][fin_y] = fid
            except KeyError:
                pass
        elif cmd == "branch":
            if new_world[action['gc_x']][action['gc_y']] == 0:
                spawn_growth_cone(new_world, dct[fid]['parent_id'], action['gc_x'], action['gc_y'], energy = 50, heading =(action['head_x'], action['head_y']) )
        elif cmd == "emit":
            head_x, head_y = action['head_x'], action['head_y']
            org_x, org_y = action['org_x'], action['org_y']
            ahead = find_fin(org_x, org_y, head_x, head_y)
            fin_x, fin_y = ahead[0], ahead[1]
            sign_x, sign_y = head_x < 0, head_y < 0
            delta_x, delta_y = 0, 0
            # go perpindicular to the direction we would've moved
            # to do this flip x and y, and invert one sign
            for i in range(0, 4):
                try:
                    if abs(head_x) > 0.5:
                        delta_y -= 1 if sign_x else -1
                    if abs(head_y) > 0.5:
                        delta_x += 1 if sign_y else -1
                    if new_world[org_x + delta_x, org_y + delta_y ] == 0:
                        spawn_dscam(new_world, dct[fid]['parent_id'], org_x + delta_x, org_y + delta_y)
                except IndexError:
                    pass

            delta_x, delta_y = 0, 0
            for i in range(0, 4):
                try:
                    if abs(head_x) > 0.5:
                        delta_y -= 1 if not sign_x else -1
                    if abs(head_y) > 0.5:
                        delta_x += 1 if not sign_y else -1
                    if new_world[org_x + delta_x, org_y + delta_y ] == 0:
                        spawn_dscam(new_world, dct[fid]['parent_id'], org_x + delta_x, org_y + delta_y)
                except IndexError:
                    pass
                    
            for i in range(0, 4):
                try:
                    if abs(head_x) > 0.5:
                        delta_x -= 1 if sign_x else -1
                    if abs(head_y) > 0.5:
                        delta_y += 1 if sign_y else -1
                    if new_world[fin_x + delta_x, fin_y + delta_y ] == 0:
                        spawn_dscam(new_world, dct[fid]['parent_id'], fin_x + delta_x, fin_y + delta_y)
                except IndexError: 
                    pass

    return new_world


# move growth cone, spawn stem 
# emit dscam
# fork growth cone
def next_state(world):
    """
    :param world:
    :return:
    """
    global dct
    new_world = np.copy(world) 
    interact_list = []
    for rows in range(1, size - 1):
        for cols in range(1, size - 1):
            try:
                focused_id = world[rows][cols]
                if dct[focused_id]['itemtype'] == "growth cone":
                    interact_list.append( interact(world, focused_id) )
                    # build up list of commands to be executed at later time
                if dct[focused_id]['itemtype'] == "dscam":
                    pass #interact_list.append( interact(world, focused_id))
            except KeyError:
                pass
    new_world = execute_actions(interact_list, new_world)
    #execute commands
    return new_world

#%%
def matrix_to_png():
    '''returns a size:size:4 matrix with corresponding colors to make image
    colors taken from dct'''
    global world
    img_world = np.zeros([size, size, 4], np.uint8)
    for rows in range(0, size - 1):
        for cols in range(0, size - 1):
            if world[rows][cols] == 0:
                img_world[rows, cols] = [255, 255, 255, 255]
            else:
                img_world[rows][cols] = dct[ world[rows][cols] ]['color']
    return img_world


# %% Use to spawn in the neurons

spawn_neuron(world, 1, (size//3, size//3))
spawn_neuron(world, 2, (size//3, size-(size//3)))
spawn_neuron(world, 3, (size-(size//3), size-(size//3)))
spawn_neuron(world, 4, (size-(size//3), size//3))

# %% Use to observe a single step of the update loop
world = next_state(world)
print(world)
pl.imshow(to_rgb(matrix_to_png()))

# %%
for i in range(0, 20):
  world = next_state(world)
pl.imshow(to_rgb(matrix_to_png()))

# %%
with VideoWriter('teaser.mp4') as vid:
    # spawn
    # spawn_neuron(world, 1, (size//3,size//3))
    # spawn_neuron(world, 2, (size-(size//3),size-(size//3)))
    # grow
    for i in tqdm.trange(500):
        world = next_state(world)
        vid.add(to_rgb(matrix_to_png()))

mvp.ipython_display('teaser.mp4', maxduration=200, loop=True)
