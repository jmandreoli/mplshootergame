from shooter import Game, GameManager
from shooter2 import Game as Game2
from collections.abc import MutableMapping

class odict (MutableMapping):
  def __init__(self,**ka): self.__dict__ = ka
  def __getitem__(self,k): return self.__dict__[k]
  def __setitem__(self,k,v): self.__dict__[k] = v
  def __delitem__(self,k): del self.__dict__[k]
  def __iter__(self): return iter(self.__dict__)
  def __len__(self): return len(self.__dict__)
  def __str__(self): return str(self.__dict__)
  def __repr__(self): return repr(self.__dict__)

def game():
  D = odict(
    fps=25,
    avatar=odict(x=.5,y=-.05,v=.125),
    targets=odict(v=.1,rate=2.,width=.02),
    bullets=odict(v=.25,rload=.4),
    hits=odict(timeout=1.),
    )
  # corrections
  D.targets.width = .04
  D.targets.rate = 1.
  D.hits.timeout = 2.
  return Game(**D)
  
def mgr():
  from matplotlib import rcParams
  rcParams['toolbar'] = 'None'

  D = odict(
    avatar=odict(c='b',marker='^'),
    targets=odict(c='r',marker='_',linewidth=5),
    bullets=odict(c='b',marker='.'),
    hits=odict(s=100,c='m',marker='*'),
    )
  # corrections

  return GameManager(**D)

mgr().play(game())

