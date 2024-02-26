mkdir graphs
cd graphs
python ../../../tools/vine_plot_compose.py ../logs/transactions --worker-view --out worker-view --sublabels 
python ../../../tools/vine_plot_compose.py ../logs/transactions --worker-cache --out worker-cache --sublabels 
python ../../../tools/vine_plot_compose.py ../logs/transactions --task-view --out task-view --sublabels 
python ../../../tools/vine_plot_compose.py ../logs/transactions --task-runtime --out task-runtime --sublabels 
python ../../../tools/vine_plot_compose.py ../logs/transactions --task-completion --out task-completion --sublabels 
python ../../../tools/vine_plot_compose.py ../logs/transactions --task-state --out task-state --sublabels 
vine_graph_log -T png ../logs/performance
mv ../logs/*.png . 

