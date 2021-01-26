   1: clear
   2: %run tree_tagger/ReadHepmc.py
   3: %refresh
   4: %reset
   5: import numpy as np
   6: clear
   7: %run tree_tagger/ReadHepmc.py
   8: %run tree_tagger/ReadHepmc.py
   9: %run tree_tagger/ReadHepmc.py
  10: import awkward
  11: awkward.load('mini.awkd')
<awkward.load (52 members)>
  12: data = awkward.load('mini.awkd')
  13: %run tree_tagger/Components
  14: data = EventWise.from_file("mini.awkd")
  15: data
<__main__.EventWise at 0x2b1c5d7d29b0>
  16: data.px
<JaggedArray [[0.0 -14.386555718774256 0.0 ... 0.14810407140722728 0.6929479260013531 0.6748763131720957] [0.0 0.0 0.6114231867883629 ... -0.12061155685441474 -0.14428968721433622 0.022466057737979134] [0.0 0.0 -4.579986643923631 ... -4.001895687671684 -1.3418528161139809 -1.4369468510779493] [0.0 0.0 -0.9710657714214195 ... -2.809674049269032 -1.4555615613140693 -1.354112487954963] [0.0 -1.0851048917420951 0.0 ... -0.1514368211044686 -0.05969310875628643 -1.3214440279857869]] at 0x2b1ca9b98358>
  17: len(data.px)
5
  18: data.Parent
  19: data.Parents
<JaggedArray [[[4] [4] [5] ... [795] [796] [796]] [[9] [11] [11] ... [1443] [1445] [1445]] [[4] [6] [6] ... [778] [779] [779]] [[115] [116] [116] ... [2118] [2130] [2130]] [[4] [4] [6] ... [2561] [2567] [2567]]] at 0x2b1ca9a475f8>
  20: data.Parents.shape
(5,)
  21: data.Parents[0][0]
array([4])
  22: data.Parents[0][4]
array([14])
  23: data.children
<JaggedArray [[[3] [7 8] [3] ... [] [] []] [[3] [3] [17] ... [] [] []] [[3] [3] [12] ... [] [] []] [[3] [3] [542 543] ... [2131 2132] [] []] [[3] [38] [3] ... [] [] []]] at 0x2b1caa296cf8>
  24: data.pT
  25: add_all(data)
  26: data.pT
<JaggedArray [[0.0 22.215180892962422 0.0 ... 0.15348537601616813 0.7481455004564337 0.7341498985394296] [0.0 0.0 11.184487669529593 ... 0.359414566386977 1.0398999373657893 0.47370454448393196] [0.0 0.0 13.305394850855116 ... 5.697026522058029 1.8683617227962557 2.0344166352725863] [0.0 0.0 3.096601274303897 ... 2.824015563868472 1.4636625180327179 1.3603829437440327] [0.0 11.077389050042942 0.0 ... 0.923865282461938 0.13354905148892865 3.82384185521331]] at 0x2b1caa1c2048>
  27: import matplotlib.pyplot as plt
  28: plt.plot(data.pT.flatten)
  29: plt.plot(data.pT.flatten())
[<matplotlib.lines.Line2D at 0x2b1cad6c5898>]
  30: plt.savefig('pT.png')
  31: plt.plot(data.is_lead.pT.flatten())
  32: plt.plot(data.is_leaf.pT.flatten())
  33: plt.plot(data.Is_leaf.pT.flatten())
  34: plt.plot(data[data.Is_leaf].pT.flatten())
  35: plt.plot(data.pT[data.Is_leaf].flatten())
[<matplotlib.lines.Line2D at 0x2b1caa2dba58>]
  36: mask = data.Is_leaf
  37: plt.plot(data.pT[mask].flatten(), data.rapidity[mask].flatten())
[<matplotlib.lines.Line2D at 0x2b1cadb8c0b8>]
  38: plt.clear()
  39: plt.cls()
  40: plt.clf()
  41: plt.plot(data.pT[mask].flatten(), data.rapidity[mask].flatten())
[<matplotlib.lines.Line2D at 0x2b1cadb91c18>]
  42: plt.savefig('pt.png')
  43: plt.plot(data.pT[mask].flatten(), data.rapidity[mask].flatten(), 'bo')
[<matplotlib.lines.Line2D at 0x2b1cadbb5c18>]
  44: plt.savefig('pt.png')
  45: plt.plot(data.pT[mask].flatten(), data.rapidity[mask].flatten(), 'o')
[<matplotlib.lines.Line2D at 0x2b1cadbde390>]
  46: plt.savefig('pt.png')
  47: plt.clf()
  48: plt.plot(data.pT[mask].flatten(), data.rapidity[mask].flatten(), 'o')
[<matplotlib.lines.Line2D at 0x2b1cadc01a90>]
  49: plt.savefig('pt.png')
  50: clear
  51: clear
  52: %history
  53: %history -o
  54: %history -o -n
  55: %history -n -o -f /home/jlc1n20/projects/particle_train/src/data_example.py
