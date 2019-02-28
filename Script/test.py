import sqlite3
import seaborn as sns

def get_data(db_file):
    db = sqlite3.connect(db_file)
    with db:
        data=pd.read_sql_query("SELECT * FROM heuristic_info",db)


######################## three factor visualization #############################
# def factor_plot():
#     sns.set(style="whitegrid")
#     sns.factorplot("sampling", "test_score", hue="balance", col="weight", data=data[data.evaluation=='auc'],hue_order=[0,1],palette="YlGnBu_d", aspect=.75).despine(left=True)
#     plt.show()

######################## bokeh interactive plot #############################
# %%output backend='bokeh'
# %%opts Bars [tools=[hover] invert_axes=True]
# p1 = data[data.sampling=='no'].hvplot.bar(
#     ['heuristic'],'test_score',groupby=['evaluation'], color='red',fill_alpha=0.1)
# (p1+p2+p3).cols(1)
# ------------------------------------------------------------------------------- #
# from bokeh.models import HoverTool
# hover = HoverTool(tooltips=[("Value", "@test_score")])
# p = data.hvplot.bar(['sampling','heuristic'],'test_score',groupby='evaluation',fill_alpha=0.2,rot=45)
# p.opts(width=800, tools=[hover],title='title',height=400,show_grid=True)