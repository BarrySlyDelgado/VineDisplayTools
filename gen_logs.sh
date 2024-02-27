mkdir graphs
cd graphs
python $HOME/VineDisplayTools/vine_plot_compose.py ../logs/transactions --worker-view --out worker-view --sublabels 
python $HOME/VineDisplayTools/vine_plot_compose.py ../logs/transactions --worker-cache --out worker-cache --sublabels 
python $HOME/VineDisplayTools/vine_plot_compose.py ../logs/transactions --task-view --out task-view --sublabels 
python $HOME/VineDisplayTools/vine_plot_compose.py ../logs/transactions --task-runtime --out task-runtime --sublabels 
python $HOME/VineDisplayTools/vine_plot_compose.py ../logs/transactions --task-completion --out task-completion --sublabels 
python $HOME/VineDisplayTools/vine_plot_compose.py ../logs/transactions --task-state --out task-state --sublabels 
python $HOME/VineDisplayTools/vine_plot_compose.py ../logs/transactions --file-accum --out file-accum --sublabels 
python $HOME/VineDisplayTools/vine_plot_compose.py ../logs/transactions --file-hist --out file-hist --sublabels 
vine_graph_log -T png -D ../logs/performance
mv ../logs/*.png . 
cp $HOME/VineDisplayTools/index.html ..

