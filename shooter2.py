__all__ = ('Targets','Bullets','Hits','Game')

import logging, os
logger = logging.getLogger(__name__)

from numpy import array, zeros, ones, empty, arange, newaxis, abs, sum, square, sqrt, log, exp, argmin, argmax, amin, amax, nonzero, all, any, nan, isnan, dot, mean, average, std
from numpy import clip, linspace, concatenate, unique, cumsum
from numpy.random import uniform, geometric

from shooter import Game as BaseGame

#----------------------------------------------------------------------------------------------------
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

   the total number of exposed sprites, which is equal to the number of frames during which any of them is exposed

.. attribute:: M

   the total number of sprites, including the non exposed ones

.. attribute:: cslice

   the lower and upper indices of the exposed part of the wave: the exposed sprites are in the slice *cslice* [0]: *cslice* [1] which is of constant length *N*

.. attribute:: xpos

   the array of horizontal positions of the sprites as a :class:`numpy.array` (( *M* ,), :const:`float` )

.. attribute:: xspeed

   the array of horizontal speeds of the sprites as a :class:`numpy.array` (( *M* ,), :const:`float` ) in x-unit/frame-transition

.. attribute:: alive

   the array of liveness status of the sprites as a :class:`numpy.array` (( *M* ,), :const:`bool` ); :const:`True` means the sprite is alive (not hit)

.. attribute:: born

   the array of birthdates (in frame number) of the sprites as a :class:`numpy.array` (( *M* ,), :const:`int` )

.. attribute:: ypos

   the array of vertical positions of the exposed sprites ONLY as a :class:`numpy.array` (( *N* ,), :const:`float` )

.. attribute:: artist

   the artist in charge of displaying the sprites
  """
#----------------------------------------------------------------------------------------------------

  def __init__(self,game,v=None,orient=None):
    self.game = game
    self.N = N = int(game.fps/v)
    self.n = 0
    self.orient = orient
    self.M = 2*N
    self.born,self.xpos,self.xspeed,self.alive = (zeros((self.M,),typ) for typ in (int,float,float,bool))
    self.ialive = zeros((N,),int)
    self.nalive = 0
    self.newcontent(t=N-1)
    self.alive[:] = True
    self.nborn = 0
    self.tborn = self.born[0]
    self.ypos = linspace(0.,1.,N)[slice(None,None,orient),newaxis]
    self.cslice = 0,N

  def update(self):
    tbeg,tend = self.cslice
    n = self.nalive
    if n:
      a = self.ialive[:n]
      s = self.alive[a]
      if self.born[a[0]] == tbeg and s[0]:
        self.leaving(a[0])
        s[0] = False
      n = sum(s)
      a[:n] = a[s]
    if self.tborn == tend:
      self.ialive[n] = self.nborn
      n += 1
      self.entering(self.nborn)
      self.nborn += 1
      if self.nborn == self.M:
        a = self.ialive[:n]
        for comp in (self.born,self.xpos,self.xspeed): comp[:n] = comp[a]
        a[:] = arange(n)
        self.nborn = n
        self.newcontent(n,tend)
        self.alive[:] = True
      self.tborn = self.born[self.nborn]
    self.nalive = n
    if n:
      a = self.ialive[:n]
      self.xpos[a] += self.xspeed[a]
    self.cslice = tbeg+1,tend+1

  def leaving(self,i): pass
  def entering(self,i): pass

  def setup(self,ax,**style):
    self.artist = ax.scatter((),(),**style)

  def display(self):
    a = self.current()
    t = self.cslice[0]
    self.artist.set_offsets(concatenate((self.xpos[a][:,newaxis],self.ypos[self.born[a]-t]),axis=1))

  def current(self):
    return self.ialive[:self.nalive]

#----------------------------------------------------------------------------------------------------
class Targets (Wave):
  """
An object of this class implements a wave of targets in a game. A new target is created stochastically, at a given rate, at the upper border of the space.

:param rate: the average number of targets created per sec
:type rate: :const:`float`
:param width: the width of a target in x-unit
:type width: :const:`float`
:param game,ka: passed to parent class

Attributes:

.. attribute:: rate

   the probability of creating a target at each new frame

.. attribute:: width

   the width of a target in x-unit

.. attribute:: score

   the cumulated number of miss
  """
#----------------------------------------------------------------------------------------------------

  def __init__(self,game,rate=None,width=None,**ka):
    self.rate = rate/game.fps
    self.width = width
    self.score = 0
    super(Targets,self).__init__(game,orient=1,**ka)

  def newcontent(self,n=0,t=-1):
    xpos,xposc = uniform(0.,1.,(2,self.M-n))
    self.xpos[n:] = xpos
    self.xspeed[n:] = (xposc-xpos)/self.N
    self.born[n:] = t+cumsum(geometric(self.rate,(self.M-n,)))

  def leaving(self,i):
    self.score += 1
    self.game.tr_miss = True

  def setup(self,ax,**style):
    super(Targets,self).setup(ax,s=self.getsize(ax),**style)

  def getsize(self,ax):
    p0,p1 = ax.transData.transform(array(((0.,0.),(1.,0.))))
    return square((p1-p0)[0]*self.width)

#----------------------------------------------------------------------------------------------------
class Bullets (Wave):
  """
An object of this class implements a wave of bullets in a game. A new bullet is created deterministically at a constant rate, at the lower border of the space.

:param rload: the time in sec between two bullets
:type rload: :const:`float`
:param game,ka: passed to parent class

Attributes:

.. attribute:: rload

   the number of frames between two consecutive bullet creations
  """
#----------------------------------------------------------------------------------------------------

  def __init__(self,game,rload=None,**ka):
    self.rload = int(rload*game.fps)
    super(Bullets,self).__init__(game,orient=-1,**ka)

  def newcontent(self,n=0,t=-1):
    self.xpos[n:] = 0.5
    self.xspeed[n:] = 0.
    self.born[n:] = t+self.rload*arange(1,self.M-n+1)

  def entering(self,i):
    self.xpos[i] = self.game.avatar.pos[0,0]

#----------------------------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------------

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
    w,w1 = self.game.targets, self.game.bullets
    a,a1 = w.current(), w1.current()
    if len(a)>0 and len(a1)>0:
      dx = w1.xpos[a1][newaxis,:] - w.xpos[a][:,newaxis]
      dv = w1.xspeed[a1][newaxis,:] - w.xspeed[a][:,newaxis]
      r,r1 = w.born[a]-w.cslice[0], w1.born[a1]-w1.cslice[0]
      m = self.clashmat[r,:][:,r1]
      nz,nz1 = nonzero((abs(dx+dv*m)<self.tol)&(m>=0)&(m<=1))
      if len(nz)>0:
        self.game.tr_hits = True
        self.score += len(nz)
        w.alive[w.ialive[nz]] = False
        w1.alive[w1.ialive[nz1]] = False
        rnz = r[nz]
        self.weight[rnz] = self.timeout
        self.xpos[rnz] = w.xpos[w.ialive[nz]]
    self.weight -= 1
    clip(self.weight,0,self.timeout,self.weight)

  def setup(self,ax,**style):
    self.artist = ax.scatter((),(),**style)

  def display(self):
    m = self.weight>0
    self.artist.set_offsets(concatenate((self.xpos[m][:,newaxis],self.ypos[m]),axis=1))

class Game (BaseGame):

  Factory = BaseGame.Factory.copy()
  Factory.update(targets=Targets,bullets=Bullets,hits=Hits)
