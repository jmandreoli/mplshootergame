__all__ = ('Avatar','Targets','Bullets','Hits','GameManager','Game')

import logging, os
logger = logging.getLogger(__name__)

from numpy import array, zeros, ones, empty, arange, newaxis, abs, sum, square, sqrt, log, exp, argmin, argmax, amin, amax, nonzero, all, any, nan, isnan, dot, mean, average, std
from numpy import clip, linspace, concatenate, unique
from numpy.random import uniform
from time import clock

from matplotlib.animation import FuncAnimation

#--------------------------------------------------------------------------------------------------
class Avatar (object):
  """
An object of this class implements the avatar in a game.

:param game: the game object
:type game: :class:`Game`
:param x: the horizontal position of the avatar in x-unit
:type x: :const:`float`
:param y: the vertical position of the avatar in y-unit
:type y: :const:`float`
:param v: the horizontal speed of the avatar in x-unit/sec
:type v: :const:`float`

Attributes:

.. attribute:: pos

   the current position of the avatar as a :class:`numpy.array` ((1,2), :const:`float` )

.. attribute:: xspeed

   the horizontal speed of the avatar as a positive :const:`float` in x-unit/frame-transition

.. attribute:: artist

   the artist in charge of displaying the avatar
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,game,x=None,y=None,v=None):
    self.game = game
    self.pos = array(((x,y),))
    self.xspeed = v/game.fps

  def update(self):
    x = self.pos[0,0:1]
    x += self.game.tr_move*self.xspeed
    clip(x,0.,1.,out=x)

  def setup(self,ax,**style):
    self.artist = ax.scatter((),(),**style)

  def display(self):
    self.artist.set_offsets(self.pos)

#--------------------------------------------------------------------------------------------------
class Wave (object):
  """
An object of this class implements a wave of sprites with constant vertical speed in a game.

:param game: the game object
:type game: :class:`Game`
:param v: vertical speed of the wave in y-unit/sec
:type v: :const:`float`
:param orient: direction of progress of the wave
:type orient: :const:int (+ or - 1)

Attributes:

.. attribute:: N

   the total number of exposed sprites, which is equal to the number of frames during which any of the sprites is exposed

.. attribute:: n

   the current index in the wave (of length 2 * *N* ): the exposed sprites are in the slice [ *n* , *n* + *N* ]

.. attribute:: orient

   the direction of progress of the wave (+/- :const:`1` )

.. attribute:: xpos

   the array of horizontal positions of the sprites as a :class:`numpy.array` ((2* *N* ,), :const:`float` )

.. attribute:: xspeed

   the array of horizontal speeds of the sprites as a :class:`numpy.array` ((2* *N* ,), :const:`float` ) in x-unit/frame-transition

.. attribute:: visible

   the array of visibility of the sprites as a :class:`numpy.array` ((2* *N* ,), :const:`bool` ); :const:`True` means the sprite is visible

.. attribute:: ypos

   the array of vertical positions of the exposed sprites ONLY as a :class:`numpy.array` (( *N* ,), :const:`float` )

.. attribute:: artist

   the artist in charge of displaying the sprites
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,game,v=None,orient=None):
    self.game = game
    self.N = N = int(game.fps/v)
    self.n = 0
    self.orient = orient
    self.xpos = zeros((2*N,),float)
    self.xspeed = zeros((2*N,),float)
    self.visible = zeros((2*N,),bool)
    self.ypos = linspace(0.,1.,N)[slice(None,None,orient),newaxis]
    self.xpos[N:], self.xspeed[N:], self.visible[N:] = self.newpage()

  def update(self):
    n,N = self.n,self.N
    m = n+N
    if self.visible[n]: self.leaving(n)
    if self.visible[m]: self.entering(m)
    self.xpos[n:m] += self.xspeed[n:m]
    n += 1
    if n==N:
      self.xpos[:N], self.xspeed[:N], self.visible[:N] = self.xpos[N:], self.xspeed[N:], self.visible[N:]
      self.xpos[N:], self.xspeed[N:], self.visible[N:] = self.newpage()
      n = 0
    self.n = n

  def setup(self,ax,**style):
    self.artist = ax.scatter((),(),**style)

  def display(self):
    xpos,xspeed,visible = self.current()
    self.artist.set_offsets(concatenate((xpos[:,newaxis],self.ypos),axis=1)[visible])

  def current(self):
    n,N = self.n,self.N
    s = slice(n,n+N)
    return self.xpos[s],self.xspeed[s],self.visible[s]

  def entering(self,n): pass
  def leaving(self,n): pass

#--------------------------------------------------------------------------------------------------
class Targets (Wave):
  """
An object of this class implements a wave of targets in a game. A new target is created at each frame, at the upper border of the space. Its visibility is randomly set. Invisible targets are not counted in hit/miss.

:param rate: the average number of targets created visible per sec
:type rate: :const:`float`
:param width: the width of a target in x-unit
:type width: :const:`float`
:param game,ka: passed to parent class

Attributes:

.. attribute:: rate

   the probability of the created target being visible at each new frame

.. attribute:: width

   the width of a target in x-unit

.. attribute:: score

   the cumulated number of miss
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,game,rate=None,width=None,**ka):
    self.rate = rate/game.fps
    self.width = width
    self.score = 0
    super(Targets,self).__init__(game,orient=1,**ka)

  def newpage(self):
    N = self.N
    xpos,xposc,visible = uniform(0.,1.,(3,N))
    xspeed = (xposc-xpos)/N
    return xpos, xspeed, visible<self.rate

  def leaving(self,n):
    self.score += 1
    self.game.tr_miss = True

  def setup(self,ax,**style):
    super(Targets,self).setup(ax,s=self.getsize(ax),**style)

  def getsize(self,ax):
    p0,p1 = ax.transData.transform(array(((0.,0.),(1.,0.))))
    return square((p1-p0)[0]*self.width)

#--------------------------------------------------------------------------------------------------
class Bullets (Wave):
  """
An object of this class implements a wave of bullets in a game. A new bullet is created at each frame, at the lower border of the space. Its visibility is set so that a fixed reload time occurs between 2 visible bullets. Invisible bullets are not counted in hit/miss.

:param rload: the time in sec between two visible bullets
:type rload: :const:`float`
:param game,ka: passed to parent class

Attributes:

.. attribute:: rload

   the number of frames between the creation of visible bullets
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,game,rload=None,**ka):
    self.rload = int(rload*game.fps)
    super(Bullets,self).__init__(game,orient=-1,**ka)

  def newpage(self):
    N = self.N
    visible = zeros(N,bool)
    visible[arange(0,N,self.rload)] = True
    return 0.5, 0., visible

  def entering(self,n):
    self.xpos[n] = self.game.avatar.pos[0,0]

#--------------------------------------------------------------------------------------------------
class Hits (object):
  """
An object of this class implements the hits (collisions target-bullet) in a game.

:param game: the game object
:type game: :class:`Game`
:param timeout: the time in sec during which a hit remains visible
:type timeout: :const:`float`

Attributes:

.. attribute:: timeout

   the number of frames during which a hit remains visible

.. attribute:: tol

   the tolerance in x-unit for a hit between a target and a bullet

.. attribute:: clashmat

   a matrix (number of targets / number of bullets) containing the time of a collision in the next frame transition (if lower than 0 or greater than 1, no collision occurs)

.. attribute:: xpos

   the array of horizontal positions of the hits in x-unit

.. attribute:: ypos

   the array of vertical positions of the hits in y-unit

.. attribute:: weight

   the array of remaining visibility duration (in number of frames) of the hits

.. attribute:: artist

   the artist in charge of displaying the hits

.. attribute:: score

   the cumulated number of hits
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,game,timeout=None):
    self.game = game
    self.timeout = int(timeout*game.fps)
    self.tol = game.targets.width/2
    r = game.targets.ypos
    N = game.targets.N
    r1 = game.bullets.ypos
    N1 = game.bullets.N
    self.clashmat = (r-r1.T)/(1./N+1./N1)
    self.ypos = r
    self.xpos = zeros((N,),float)
    self.weight = zeros((N,),int)
    self.score = 0

  def update(self):
    x,v,visible = self.game.targets.current()
    x1,v1,visible1 = self.game.bullets.current()
    if any(visible) and any(visible1):
      dx = x1[visible1][newaxis,:]-x[visible][:,newaxis]
      dv = v1[visible1][newaxis,:]-v[visible][:,newaxis]
      m = self.clashmat[visible,:][:,visible1]
      nz,nz1 = nonzero((abs(dx+dv*m)<self.tol)&(m>=0.)&(m<=1.))
      if len(nz)>0:
        self.game.tr_hits = True
        self.score += len(nz)
        s = nonzero(visible)[0][nz]
        visible[s] = False
        self.weight[s] = self.timeout
        self.xpos[s] = x[s]
        visible1[nonzero(visible1)[0][nz1]] = False
    self.weight -= 1
    clip(self.weight,0,self.timeout,self.weight)

  def setup(self,ax,**style):
    self.artist = ax.scatter((),(),**style)

  def display(self):
    m = self.weight>0
    self.artist.set_offsets(concatenate((self.xpos[m][:,newaxis],self.ypos[m]),axis=1))

#--------------------------------------------------------------------------------------------------
class Game (object):
  """
An object of this class implements the full functionality of the game with its components.

:param fps: the number of frames per second
:param avatar: configuration of the avatar component
:type avatar: :const:`dict`
:param targets: configuration of the targets component
:type targets: :const:`dict`
:param bullets: configuration of the bullets component
:type bullets: :const:`dict`
:param hits: configuration of the hits component
:type hits: :const:`dict`

.. attribute:: fps

   the number of frames per second

.. attribute:: tr_move, tr_quit

   the user input for a frame transition; :attr:`tr_move` : user move command in -2,-1,0,1,2; :attr:`tr_quit` : user quit command :const:`bool`

.. attribute:: tr_hits, tr_miss

   the user output; :attr:`tr_hits` : whether there were hits in the frame transition; :attr:``tr_miss` : whether there were miss in the frame transition

.. attribute:: nstep

   the number of frame transitions performed so far

.. attribute:: gameover

   whether the game is over

.. attribute:: status

   the status text as appears in the status bar

.. attribute:: perf

   a performance measure (for profiling)

.. attribute:: components

   the component objects of the game (avatar, targets, bullets, hits)
  """
#--------------------------------------------------------------------------------------------------

  Factory = dict(avatar=Avatar,targets=Targets,bullets=Bullets,hits=Hits)

  def __init__(self,fps=None,**config):
    self.fps = fps
    self.nstep = 0
    self.gameover = False
    self.status = 'time: 0'
    self.perf = 0.
    self.components = []
    for cn in ('avatar','targets','bullets','hits'):
      c = self.Factory[cn](self,**config[cn])
      self.components.append((cn,c))
      setattr(self,cn,c)

  def update(self):
    start = clock()
    if self.tr_quit:
      self.gameover = True
      self.status = 'Game over. Efficiency (logic): {:.2%}'.format(self.perf*self.fps)
    else:
      for cn,c in self.components: c.update()
      hit = self.hits.score
      miss = self.targets.score
      total = miss+hit
      score = '{:.2f}'.format(100*float(hit)/float(total)) if total else '*'
      self.nstep += 1
      self.status = 'time: {:06.1f}; hit: {}; miss: {}; score: {}'.format(self.nstep/self.fps,hit,miss,score)
      self.perf += (clock()-start-self.perf)/self.nstep

  def setup(self,mgr):
    def loop():
      while not self.gameover:
        yield
        mgr.userinput(self)
        self.update()
        mgr.useroutput(self)
      yield
    ax = mgr.figure.add_axes((0,0,1,1),xticks=(),yticks=())
    ax.set_xlim(0.,1.)
    ax.set_ylim(-.1,1.)
    ax.axhline(0.,c='k')
    self.a_status = ax.text(.01,-.09,'*',fontsize='xx-small',bbox=dict(edgecolor='k',facecolor='none'))
    for cn,c in self.components: c.setup(ax,**mgr.config[cn])
    F = lambda *a: self.display()
    self.anim = FuncAnimation(mgr.figure,frames=loop,interval=1000/self.fps,func=F)

  def display(self):
    self.a_status.set_text(self.status)
    for cn,c in self.components: c.display()

#--------------------------------------------------------------------------------------------------
class GameManager (object):
  """
An object of this class is in charge of managing interaction with the user, using keyboard only.

Attributes

.. attribute:: figure

   A :class:`matplotlib.figure` object supporting the game

.. attribute:: keys

   A bit-vector (int) holding the keys which have been pressed and not released yet; each key of interest is assigned a bit position
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,soundpath=os.path.join(os.path.dirname(__file__),'sound'),**config):
    try: from winsound import PlaySound, SND_ASYNC, SND_FILENAME
    except: self.usernotify = lambda sid: None
    else:
      D = dict((os.path.splitext(x)[0],os.path.join(soundpath,x)) for x in os.listdir(soundpath))
      self.usernotify = lambda sid: PlaySound(D[sid],SND_ASYNC|SND_FILENAME)
    self.config = config

  def play(self,game):
    """
Plays one *game*

:param game: the game to play
:type game: :class:`Game`
    """
    from matplotlib.pyplot import figure, show
    self.keys = 0
    self.figure = fig = figure()
    def kpress(ev): self.keys |= KEYS.get(ev.key.rsplit('+',1)[-1],0)
    def krelease(ev): self.keys &= ~KEYS.get(ev.key.rsplit('+',1)[-1],0)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event',kpress)
    fig.canvas.mpl_connect('key_release_event',krelease)
    game.setup(self)
    show()

  def userinput(self,game):
    """
Turns the user input (key controls) into input for a transition of *game* .

:param game: the game to play
:type game: :class:`Game`
    """
    m = bool(self.keys&KEY_MOVERIGHT)-bool(self.keys&KEY_MOVELEFT)
    if self.keys&KEY_BOOST: m *= 2
    game.tr_move = m
    game.tr_quit = bool(self.keys & KEY_TERMINATE)
    game.tr_hits = False
    game.tr_miss = False

  def useroutput(self,game):
    """
Turns the output of a transition of *game* into user output (sound).

:param game: the game to play
:type game: :class:`Game`
    """
    if game.tr_hits: self.usernotify('hits')
    if game.tr_miss: self.usernotify('miss')

KEY_MOVELEFT = 1
KEY_MOVERIGHT = 2
KEY_BOOST = 4
KEY_TERMINATE = 8
KEYS = {'left':KEY_MOVELEFT, 'right':KEY_MOVERIGHT, 'control':KEY_BOOST, 'alt':KEY_BOOST, ' ':KEY_TERMINATE}
