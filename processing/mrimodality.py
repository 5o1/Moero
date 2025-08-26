from enum import Enum, auto

class MriModality:
    name = "mrimodality"
    def __str__(self):
        return self.__class__.name

class Mapping(MriModality):
    name = "mapping"

class T1map(Mapping, MriModality):
    name = "t1map"

class T2map(Mapping, MriModality):
    name = "t2map"

class T2smap(Mapping, MriModality):
    name = "t2smap"

class T1mappost(Mapping, MriModality):
    name = "t1mappost"

class BlackBlood(MriModality):
    name = "blackblood"

class T1w(MriModality):
    name = "t1w"

class T2w(MriModality):
    name = "t2w"

class T1rho(MriModality):
    name = "t1rho"

class LGE(MriModality):
    name = "lge"

class Flow2d(MriModality):
    name = "flow2d"

class Cine(MriModality):
    name = "cine"

class Perfusion(MriModality):
    name = "perfusion"