let SessionLoad = 1
if &cp | set nocp | endif
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/projects/jetTools
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
argglobal
%argdel
$argadd ~/projects/particle_train/src/data/hepmc2hdf5.py
edit ~/projects/particle_train/src/data/io.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 90 + 136) / 272)
exe 'vert 2resize ' . ((&columns * 90 + 136) / 272)
exe 'vert 3resize ' . ((&columns * 90 + 136) / 272)
argglobal
balt ../particle_train/hepwork/data/io.py
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=99
setlocal fml=1
setlocal fdn=20
setlocal fen
12
normal! zo
16
normal! zo
26
normal! zo
27
normal! zo
33
normal! zo
42
normal! zo
52
normal! zo
58
normal! zo
68
normal! zo
76
normal! zo
77
normal! zo
87
normal! zo
let s:l = 30 - ((20 * winheight(0) + 19) / 39)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 30
normal! 015|
wincmd w
argglobal
if bufexists("~/projects/particle_train/src/data/hepmc2hdf5.py") | buffer ~/projects/particle_train/src/data/hepmc2hdf5.py | else | edit ~/projects/particle_train/src/data/hepmc2hdf5.py | endif
balt jet_tools/src/ReadHepmc.py
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=99
setlocal fml=1
setlocal fdn=20
setlocal fen
23
normal! zo
48
normal! zo
49
normal! zo
71
normal! zo
72
normal! zo
99
normal! zo
102
normal! zo
155
normal! zo
let s:l = 108 - ((12 * winheight(0) + 19) / 39)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 108
normal! 0
wincmd w
argglobal
if bufexists("~/projects/lgn/LorentzGroupNetwork/data/jet/root2h5/root2h5.py") | buffer ~/projects/lgn/LorentzGroupNetwork/data/jet/root2h5/root2h5.py | else | edit ~/projects/lgn/LorentzGroupNetwork/data/jet/root2h5/root2h5.py | endif
balt ~/projects/lgn/LorentzGroupNetwork/data/jet/root2h5
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=99
setlocal fml=1
setlocal fdn=20
setlocal fen
16
normal! zo
38
normal! zo
59
normal! zo
62
normal! zo
95
normal! zo
let s:l = 1 - ((0 * winheight(0) + 19) / 39)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1
normal! 0
lcd ~/projects/particle_train
wincmd w
exe 'vert 1resize ' . ((&columns * 90 + 136) / 272)
exe 'vert 2resize ' . ((&columns * 90 + 136) / 272)
exe 'vert 3resize ' . ((&columns * 90 + 136) / 272)
tabnext 1
badd +71 ~/projects/particle_train/src/data/hepmc2hdf5.py
badd +1 ~/projects/particle_train/scripts/gen_event.sh
badd +59 ~/projects/jetTools/jet_tools/src/ReadHepmc.py
badd +3653 ~/projects/jetTools/jet_tools/src/FormJets.py
badd +1 ~/projects/lgn/LorentzGroupNetwork/data/jet/root2h5/root2h5.py
badd +1 ~/projects/lgn/LorentzGroupNetwork/data/jet/root2h5
badd +0 ~/projects/particle_train/scripts/test.sh
badd +1 ~/projects/jetTools/jet_tools/src/TrueTag.py
badd +1 ~/projects/lgn/LorentzGroupNetwork/data/jet/root2h5/jet_selector.py
badd +28 ~/.vimrc
badd +16 ~/projects/particle_train/__doc__
badd +3 ~/projects/jetTools/jet_tools/check_data.py
badd +5 ~/projects/jetTools/jet_tools/src/AreaMeasures.py
badd +12 ~/projects/jetTools/jet_tools/src/Components.py
badd +59 ~/projects/particle_train/src/data_example.py
badd +1 ~/projects/particle_train/src/data/make_dataset.py
badd +1 ~/projects/particle_train/src/data/load_root.py
badd +10 ~/projects/lgn/data_explore.py
badd +589 ~/.ipython/profile_default/ipython_config.py
badd +5 ~/projects/lgn/scripts/subtest.sh
badd +26 /mainfs/scratch/jlc1n20/data/toptag/Cards/run_card.dat
badd +5 ~/projects/particle_train/gen_event.sh
badd +28 ~/projects/particle_train/scripts/chain.sh
badd +26 /mainfs/scratch/jlc1n20/data/toptag/Cards/run_card_default.dat
badd +1 ~/projects/particle_train/test.sh
badd +25 ~/projects/particle_train/scripts/gen_one.sh
badd +17 ~/projects/jetTools/jet_tools/src/JoinHepMCRoot.py
badd +0 ~/projects/particle_train/src/data/io.py
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOS
set winminheight=1 winminwidth=1
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
nohlsearch
let g:this_session = v:this_session
let g:this_obsession = v:this_session
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
