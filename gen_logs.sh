mkdir graphs
cd graphs
python ../../../VineDisplayTools/vine_plot_compose.py ../logs/transactions --worker-view --out worker-view --sublabels 
python ../../../VineDisplayTools/vine_plot_compose.py ../logs/transactions --worker-cache --out worker-cache --sublabels 
python ../../../VineDisplayTools/vine_plot_compose.py ../logs/transactions --task-view --out task-view --sublabels 
python ../../../VineDisplayTools/vine_plot_compose.py ../logs/transactions --task-runtime --out task-runtime --sublabels 
python ../../../VineDisplayTools/vine_plot_compose.py ../logs/transactions --task-completion --out task-completion --sublabels 
python ../../../VineDisplayTools/vine_plot_compose.py ../logs/transactions --task-state --out task-state --sublabels 
python ../../../VineDisplayTools/vine_plot_compose.py ../logs/transactions --file-accum --out file-accum --sublabels 
python ../../../VineDisplayTools/vine_plot_compose.py ../logs/transactions --file-hist --out file-hist --sublabels 
vine_graph_log -T png ../logs/performance
mv ../logs/*.png . 

