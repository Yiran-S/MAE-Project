#!/usr/bin/env python
# coding: utf-8

# <h1>ECON 435 Final Project<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Task1:-Quantify-the-company’s-growth" data-toc-modified-id="Task1:-Quantify-the-company’s-growth-1">Task1: Quantify the company’s growth</a></span></li><li><span><a href="#Task2:-Identify-the-source-company’s-growth" data-toc-modified-id="Task2:-Identify-the-source-company’s-growth-2">Task2: Identify the source company’s growth</a></span></li><li><span><a href="#Task3:-Create-conversion-funnels-for-major-sources" data-toc-modified-id="Task3:-Create-conversion-funnels-for-major-sources-3">Task3: Create conversion funnels for major sources</a></span><ul class="toc-item"><li><span><a href="#1st:-Create-a-temporary-table-to-store-if-each-customer-ever-make-it-to-the-each-kind-of-page" data-toc-modified-id="1st:-Create-a-temporary-table-to-store-if-each-customer-ever-make-it-to-the-each-kind-of-page-3.1">1st: Create a temporary table to store if each customer ever make it to the each kind of page</a></span></li><li><span><a href="#2nd:-Output-of-Gsearch-and-Bsearch-with-quantity-and-rate" data-toc-modified-id="2nd:-Output-of-Gsearch-and-Bsearch-with-quantity-and-rate-3.2">2nd: Output of Gsearch and Bsearch with quantity and rate</a></span></li><li><span><a href="#3rd:-Output-of-Homepage-and-Custom_lander-under-Gsearch-with-Quantity-and-Rate" data-toc-modified-id="3rd:-Output-of-Homepage-and-Custom_lander-under-Gsearch-with-Quantity-and-Rate-3.3">3rd: Output of Homepage and Custom_lander under Gsearch with Quantity and Rate</a></span></li></ul></li></ul></div>

# **<center>Authors<center>**
# **<center>Master of Quantitative Economics<center>**
# <center>Yiran Sun (905629996)<center>
# <center>Pingshun Xin (305642750)<center>
# <center>Yiheng An (805640602)<center> 
# <center>Xinyi Zhang (805641673)<center>
# 
# 

# In[1]:


# Import packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import pandas as pd
from plotly.figure_factory import create_table
import plotly.express as px
import pymysql


# In[2]:


# Run sql in python
def run_sql(sql,result = True):
    
    # build connection
    try:
        db = pymysql.connect(host= '127.0.0.1', port= 3306, user= 'root', password= '******', db= 'mavenfuzzyfactory')
    except:
        print('Connection failure，try it again :-)')
    
    if result == True:
        
        # prepare a cursor object 
        cursor = db.cursor()
        count = cursor.execute(sql)
        print('%d rows successfully gathered ヽ(✿ﾟ▽ﾟ)ノ'%count)
    
        # fetch results
        results = cursor.fetchall()
        df = pd.DataFrame(list(results))
    
        # fetch column names
        col = list(cursor.description)
        name = []
        for i in range(len(col)):
            name.append(list(cursor.description)[i][0])
        df.columns = name
        return df
    
    else:
        cursor = db.cursor()
        cursor.execute(sql)
        print("This operation has no need to return results.")        
    
    # close connection
    db.close()


# ### Task1: Quantify the company’s growth

# In[3]:


# sessions and orders in each month
sql = """
SELECT 
    created_yr AS yr,
    created_month AS mo, 
    COUNT(DISTINCT website_sessions.website_session_id) AS sessions, 
    COUNT(DISTINCT orders.order_id) AS orders, 
    COUNT(DISTINCT orders.order_id)/COUNT(DISTINCT website_sessions.website_session_id) AS conv_rate
FROM website_sessions
LEFT JOIN orders 
ON orders.website_session_id = website_sessions.website_session_id
GROUP BY 1,2;
"""

df1 = run_sql(sql)
df1.head(10)
# monthly sessions/orders from 2012-2014 to show the sessions and orders are gowring by month. 
# And the conversion rate is fluctuating but overally showing a incresing trend.
# Our question would be: What brings this increase? 
# So we want to break it down to source first 


# In[4]:


# Visualization: Monthly trends for gsearch sessions and orders¶
px.bar(df1, x='mo', y='sessions',
       hover_data=['orders', 'sessions'], color='orders',
       animation_frame="yr", animation_group="mo", log_x=False,
       range_y=[0,30000],height=500)  # dictionary


# ### Task2: Identify the source company’s growth

# In[5]:


# sessions and orders from each traffic source
sql = """
SELECT utm_source, 
COUNT(DISTINCT website_sessions.website_session_id) AS sessions,
COUNT(DISTINCT orders.order_id) AS orders,
COUNT(DISTINCT orders.order_id)/COUNT(DISTINCT website_sessions.website_session_id) AS conv_rate 
FROM website_sessions
LEFT JOIN orders 
ON website_sessions.website_session_id = orders.website_session_id
GROUP BY utm_source;
"""

df2 = run_sql(sql)
df2
# 'gsearch' seems to be the biggest driver of business. 
# And we would like to see what would happen if we group by source and campaign


# In[6]:


# Visualization: bar chart
fig = px.bar(df2, x='utm_source', y='sessions',
             hover_data=['orders', 'conv_rate'], color='orders') 
fig.show()


# In[7]:


# revenue from each source and campaign
sql = """
SELECT utm_source, utm_campaign, 
COUNT(DISTINCT website_sessions.website_session_id) AS sessions,
COUNT(DISTINCT orders.order_id) AS orders,
COUNT(DISTINCT orders.order_id)/COUNT(DISTINCT website_sessions.website_session_id) AS conv_rate 
FROM website_sessions LEFT JOIN orders 
ON website_sessions.website_session_id=orders.website_session_id
GROUP BY utm_source,utm_campaign
ORDER BY sessions DESC, orders DESC;
"""
df3=run_sql(sql)
df3

# 'gsearch' and 'nonbrand' under it seems to be the biggest driver of business. 
# however we can observe the conversion rate is highest under 'bsearch' and 'brand'
# So we will compare this two source conversion funnel to see which part is having issue in the following steps.


# In[8]:


# Visualization: bar chart
fig = px.bar(df3, x='utm_source', y='sessions',hover_data=['orders', 'conv_rate'], color='orders') 
fig.show()

#import plotly.graph_objects as go

#fig = go.Figure()
#fig.add_trace(go.Histogram(histfunc="sum", y=df3['sessions'], x=df3['utm_source'], name="sum"))
#fig.add_trace(go.Histogram(histfunc="sum", y=df3['orders'], x=df3['utm_campaign'], name="sum"))
#fig.show()


# In[9]:


# check the pageview type, and find the 1st page view 
sql = """
SELECT pageview_url, COUNT(pageview_url)
FROM website_pageviews
GROUP BY pageview_url; 
"""
df4=run_sql(sql)
df4


# In[10]:


# Visualization: bar chart
fig = px.bar(df4, x='pageview_url', y='COUNT(pageview_url)') 
fig.show()


# In[11]:


# we might want to take a look at what are the first pages people view
sql = """
SELECT pageview_url, COUNT(pageview_url) AS num_pageview
FROM website_pageviews 
WHERE (website_pageview_id) in 

(SELECT MIN(website_pageview_id) 
FROM website_pageviews 
GROUP BY website_session_id)

GROUP BY pageview_url
ORDER BY COUNT(pageview_url) DESC;
"""
df5=run_sql(sql)
df5
# And we can see that there are six kinds of pages that people will view at first, one is from home page 
#and all other five are from custom lander


# In[12]:


fig = px.bar(df5, x='pageview_url', y='num_pageview') 
fig.show()


# In[13]:


# Check customer from 'gsearch and nonbrand' where its customer come from
sql = """
SELECT DISTINCT pageview_url
FROM website_pageviews 
WHERE (website_pageview_id) in 
(SELECT MIN(website_pageviews.website_pageview_id) 
FROM website_pageviews 
LEFT JOIN website_sessions ON website_sessions.website_session_id = website_pageviews.website_session_id
WHERE website_sessions.utm_source = 'gsearch' AND website_sessions.utm_campaign = 'nonbrand'
GROUP BY website_pageviews.website_session_id);
"""
run_sql(sql)


# In[14]:


#Check customer from 'bsearch and brand' where its customer come from
sql = """
SELECT DISTINCT pageview_url
FROM website_pageviews 
WHERE (website_pageview_id) in 
(SELECT MIN(website_pageviews.website_pageview_id) 
 FROM website_pageviews 
 LEFT JOIN website_sessions ON website_sessions.website_session_id = website_pageviews.website_session_id
 WHERE website_sessions.utm_source = 'bsearch' AND website_sessions.utm_campaign = 'brand'
 GROUP BY website_pageviews.website_session_id);
"""
run_sql(sql)


# ### Task3: Create conversion funnels for major sources

# In[15]:


# check an entire order, from clicking into the website to finish the order
sql = """
SELECT * FROM website_sessions ws
JOIN website_pageviews wp ON ws.website_session_id=wp.website_session_id
JOIN orders o ON wS.website_session_id=o.website_session_id
WHERE ws.website_session_id=909;
"""
df6=run_sql(sql)
df6


# In[ ]:





# #### 1st: Create a temporary table to store if each customer ever make it to the each kind of page

# In[16]:


# 1st: create a temporary table to store if each customer ever make it to the each kind of page
sql = """
SELECT * FROM 
(
SELECT          # the temporary table session_level_made_it
	website_session_id, created_yr, created_month, utm_source, utm_campaign,
    MAX(homepage) AS saw_homepage, 
    MAX(custom_lander) AS saw_custom_lander,
    MAX(products_page) AS product_made_it, 
    MAX(intro_page) AS intro_made_it, 
    MAX(cart_page) AS cart_made_it,
    MAX(shipping_page) AS shipping_made_it,
    MAX(billing_page) AS billing_made_it,
    MAX(thankyou_page) AS thankyou_made_it
FROM(
SELECT
	website_sessions.website_session_id, website_pageviews.pageview_url, 
    website_sessions.created_yr, website_sessions.created_month,
    website_sessions.utm_source, website_sessions.utm_campaign,
    CASE WHEN pageview_url = '/home' THEN 1 ELSE 0 END AS homepage,
    CASE WHEN (pageview_url = '/lander-1' OR pageview_url = '/lander-2' OR pageview_url = '/lander-3' OR 
              pageview_url = '/lander-4' OR pageview_url = '/lander-5') THEN 1 ELSE 0 END AS custom_lander,
    CASE WHEN pageview_url = '/products' THEN 1 ELSE 0 END AS products_page,
    CASE WHEN (pageview_url = '/the-original-mr-fuzzy' OR pageview_url = '/the-forever-love-bear' OR 
			  pageview_url = '/the-birthday-sugar-panda' OR  pageview_url = '/the-hudson-river-mini-bear') THEN 1 
              ELSE 0 END AS intro_page, 
    CASE WHEN pageview_url = '/cart' THEN 1 ELSE 0 END AS cart_page,
    CASE WHEN pageview_url = '/shipping' THEN 1 ELSE 0 END AS shipping_page,
    CASE WHEN (pageview_url = '/billing' OR pageview_url = '/billing-2') THEN 1 ELSE 0 END AS billing_page,
    CASE WHEN pageview_url = '/thank-you-for-your-order' THEN 1 ELSE 0 END AS thankyou_page
FROM website_sessions 
	LEFT JOIN website_pageviews 
		ON website_sessions.website_session_id = website_pageviews.website_session_id
WHERE (website_sessions.utm_source = 'gsearch' AND website_sessions.utm_campaign = 'nonbrand') OR 
      (website_sessions.utm_source = 'bsearch' AND website_sessions.utm_campaign = 'brand')
ORDER BY 
	website_sessions.website_session_id,
    website_pageviews.created_at) AS pageview_level
GROUP BY 
	website_session_id) AS session_level_made_it
LIMIT 5;

"""
df7=run_sql(sql)
df7


# In[ ]:





# #### 2nd: Output of Gsearch and Bsearch with quantity and rate

# In[17]:


# 2nd: output 1st result: with quantity and rate for two sources
sql = """

SELECT utm_source, utm_campaign,
    COUNT(DISTINCT website_session_id) AS sessions,
    COUNT(DISTINCT CASE WHEN product_made_it = 1 THEN website_session_id ELSE NULL END) AS to_products,
    COUNT(DISTINCT CASE WHEN intro_made_it = 1 THEN website_session_id ELSE NULL END) AS to_intro,
    COUNT(DISTINCT CASE WHEN cart_made_it = 1 THEN website_session_id ELSE NULL END) AS to_cart,
    COUNT(DISTINCT CASE WHEN shipping_made_it = 1 THEN website_session_id ELSE NULL END) AS to_shipping,
    COUNT(DISTINCT CASE WHEN billing_made_it = 1 THEN website_session_id ELSE NULL END) AS to_billing,
    COUNT(DISTINCT CASE WHEN thankyou_made_it = 1 THEN website_session_id ELSE NULL END) AS to_thankyou,
    COUNT(DISTINCT CASE WHEN product_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT website_session_id) AS lander_products_rt,
    COUNT(DISTINCT CASE WHEN intro_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT CASE WHEN product_made_it = 1 THEN website_session_id ELSE NULL END) AS products_intro_rt,
    COUNT(DISTINCT CASE WHEN cart_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT CASE WHEN intro_made_it = 1 THEN website_session_id ELSE NULL END) AS intro_cart_rt,
	COUNT(DISTINCT CASE WHEN shipping_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT CASE WHEN cart_made_it = 1 THEN website_session_id ELSE NULL END) AS cart_ship_rt,
    COUNT(DISTINCT CASE WHEN billing_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT CASE WHEN shipping_made_it = 1 THEN website_session_id ELSE NULL END) AS ship_bill_rt,
    COUNT(DISTINCT CASE WHEN thankyou_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT CASE WHEN billing_made_it = 1 THEN website_session_id ELSE NULL END) AS bill_thankyou_rt
FROM (

# the temporary table session_level_made_it
SELECT 
	website_session_id, created_yr, created_month, utm_source, utm_campaign,
    MAX(homepage) AS saw_homepage, 
    MAX(custom_lander) AS saw_custom_lander,
    MAX(products_page) AS product_made_it, 
    MAX(intro_page) AS intro_made_it, 
    MAX(cart_page) AS cart_made_it,
    MAX(shipping_page) AS shipping_made_it,
    MAX(billing_page) AS billing_made_it,
    MAX(thankyou_page) AS thankyou_made_it
FROM
(
SELECT
	website_sessions.website_session_id, website_pageviews.pageview_url, 
    website_sessions.created_yr, website_sessions.created_month,
    website_sessions.utm_source, website_sessions.utm_campaign,
    CASE WHEN pageview_url = '/home' THEN 1 ELSE 0 END AS homepage,
    CASE WHEN (pageview_url = '/lander-1' OR pageview_url = '/lander-2' OR pageview_url = '/lander-3' OR 
              pageview_url = '/lander-4' OR pageview_url = '/lander-5') THEN 1 ELSE 0 END AS custom_lander,
    CASE WHEN pageview_url = '/products' THEN 1 ELSE 0 END AS products_page,
    CASE WHEN (pageview_url = '/the-original-mr-fuzzy' OR pageview_url = '/the-forever-love-bear' OR 
			  pageview_url = '/the-birthday-sugar-panda' OR  pageview_url = '/the-hudson-river-mini-bear') THEN 1 
              ELSE 0 END AS intro_page, 
    CASE WHEN pageview_url = '/cart' THEN 1 ELSE 0 END AS cart_page,
    CASE WHEN pageview_url = '/shipping' THEN 1 ELSE 0 END AS shipping_page,
    CASE WHEN (pageview_url = '/billing' OR pageview_url = '/billing-2') THEN 1 ELSE 0 END AS billing_page,
    CASE WHEN pageview_url = '/thank-you-for-your-order' THEN 1 ELSE 0 END AS thankyou_page
FROM website_sessions 
	LEFT JOIN website_pageviews 
		ON website_sessions.website_session_id = website_pageviews.website_session_id
WHERE (website_sessions.utm_source = 'gsearch' AND website_sessions.utm_campaign = 'nonbrand') OR 
      (website_sessions.utm_source = 'bsearch' AND website_sessions.utm_campaign = 'brand')
ORDER BY 
	website_sessions.website_session_id,
    website_pageviews.created_at) AS pageview_level
GROUP BY 
	website_session_id) AS session_level_made_it

GROUP BY 1,2;
"""
df8=run_sql(sql)
df8


# In[44]:



fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'Gsearch',
    orientation = "h",
    y = ["sessions", "to_products", "to_mrfuzzy", "to_cart","to_shipping", "to_billing", "to_thankyou"],
    x = [247564, 134519, 106502, 47653, 32186, 25961, 15797],
    textposition = "inside",
    marker = {"color":  ['rgb(222,235,247)', 'rgb(198,219,239)', 'rgb(158,202,225)', 'rgb(107,174,214)', 'rgb(39,116,174)', 'rgb(0,85,135)', 'rgb(0,59,92)']},
    textinfo = "value+percent previous",textfont_size=13))

fig.add_trace(go.Funnel(
    name = 'Bsearch',
    y = ["sessions", "to_products", "to_mrfuzzy", "to_cart","to_shipping", "to_billing", "to_thankyou"],
    x = [6545,4202,3472,1569,1091,897,564],
    marker = {"color":  ['rgb(255,247,188)', 'rgb(254,227,145)', 'rgb(255,199,44)', 'rgb(255,184,28)', 'rgb(255,184,28)', 'rgb(255,184,28)', 'rgb(255,184,28)']},
    textinfo = "value+percent previous",textfont_size=13))

fig.update_layout( height=650, width=900, 
                  title_text="Conversion Funnel for Gsearch and Bsearch",
                 font_size=15)
fig.show()


# #### 3rd: Output of Homepage and Custom_lander under Gsearch with Quantity and Rate

# In[19]:


#3rd: output 2nd result: with quantity and rate for two segment under gsearch&nonbrand
sql = """
SELECT utm_source, utm_campaign,
	CASE 
		WHEN saw_homepage = 1 THEN 'homepage'
        WHEN saw_custom_lander = 1 THEN 'custom_lander'
        ELSE 'uh oh... check logic' 
	END AS segment, 
    COUNT(DISTINCT website_session_id) AS sessions,
    COUNT(DISTINCT CASE WHEN product_made_it = 1 THEN website_session_id ELSE NULL END) AS to_products,
    COUNT(DISTINCT CASE WHEN intro_made_it = 1 THEN website_session_id ELSE NULL END) AS to_intro,
    COUNT(DISTINCT CASE WHEN cart_made_it = 1 THEN website_session_id ELSE NULL END) AS to_cart,
    COUNT(DISTINCT CASE WHEN shipping_made_it = 1 THEN website_session_id ELSE NULL END) AS to_shipping,
    COUNT(DISTINCT CASE WHEN billing_made_it = 1 THEN website_session_id ELSE NULL END) AS to_billing,
    COUNT(DISTINCT CASE WHEN thankyou_made_it = 1 THEN website_session_id ELSE NULL END) AS to_thankyou,
	COUNT(DISTINCT CASE WHEN product_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT website_session_id) AS lander_products_rt,
    COUNT(DISTINCT CASE WHEN intro_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT CASE WHEN product_made_it = 1 THEN website_session_id ELSE NULL END) AS products_intro_rt,
    COUNT(DISTINCT CASE WHEN cart_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT CASE WHEN intro_made_it = 1 THEN website_session_id ELSE NULL END) AS intro_cart_rt,
    COUNT(DISTINCT CASE WHEN shipping_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT CASE WHEN cart_made_it = 1 THEN website_session_id ELSE NULL END) AS cart_ship_rt,
    COUNT(DISTINCT CASE WHEN billing_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT CASE WHEN shipping_made_it = 1 THEN website_session_id ELSE NULL END) AS ship_bill_rt,
    COUNT(DISTINCT CASE WHEN thankyou_made_it = 1 THEN website_session_id ELSE NULL END)/COUNT(
    DISTINCT CASE WHEN billing_made_it = 1 THEN website_session_id ELSE NULL END) AS bill_thankyou_rt
FROM

# the temporary table session_level_made_it
(
SELECT 
	website_session_id, created_yr, created_month, utm_source, utm_campaign,
    MAX(homepage) AS saw_homepage, 
    MAX(custom_lander) AS saw_custom_lander,
    MAX(products_page) AS product_made_it, 
    MAX(intro_page) AS intro_made_it, 
    MAX(cart_page) AS cart_made_it,
    MAX(shipping_page) AS shipping_made_it,
    MAX(billing_page) AS billing_made_it,
    MAX(thankyou_page) AS thankyou_made_it
FROM
(
SELECT
	website_sessions.website_session_id, website_pageviews.pageview_url, 
    website_sessions.created_yr, website_sessions.created_month,
    website_sessions.utm_source, website_sessions.utm_campaign,
    CASE WHEN pageview_url = '/home' THEN 1 ELSE 0 END AS homepage,
    CASE WHEN (pageview_url = '/lander-1' OR pageview_url = '/lander-2' OR pageview_url = '/lander-3' OR 
              pageview_url = '/lander-4' OR pageview_url = '/lander-5') THEN 1 ELSE 0 END AS custom_lander,
    CASE WHEN pageview_url = '/products' THEN 1 ELSE 0 END AS products_page,
    CASE WHEN (pageview_url = '/the-original-mr-fuzzy' OR pageview_url = '/the-forever-love-bear' OR 
			  pageview_url = '/the-birthday-sugar-panda' OR  pageview_url = '/the-hudson-river-mini-bear') THEN 1 
              ELSE 0 END AS intro_page, 
    CASE WHEN pageview_url = '/cart' THEN 1 ELSE 0 END AS cart_page,
    CASE WHEN pageview_url = '/shipping' THEN 1 ELSE 0 END AS shipping_page,
    CASE WHEN (pageview_url = '/billing' OR pageview_url = '/billing-2') THEN 1 ELSE 0 END AS billing_page,
    CASE WHEN pageview_url = '/thank-you-for-your-order' THEN 1 ELSE 0 END AS thankyou_page
FROM website_sessions 
	LEFT JOIN website_pageviews 
		ON website_sessions.website_session_id = website_pageviews.website_session_id
WHERE (website_sessions.utm_source = 'gsearch' AND website_sessions.utm_campaign = 'nonbrand') OR 
      (website_sessions.utm_source = 'bsearch' AND website_sessions.utm_campaign = 'brand')
ORDER BY 
	website_sessions.website_session_id,
    website_pageviews.created_at) AS pageview_level
GROUP BY 
	website_session_id) AS session_level_made_it
WHERE utm_source='gsearch'
GROUP BY 1,2,3;

"""

df9=run_sql(sql)
df9


# In[49]:



fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'Gsearch-homepage',
    orientation = "h",
    y = ["sessions", "to_products", "to_mrfuzzy", "to_cart","to_shipping", "to_billing", "to_thankyou"],
    x = [234184,129173,102654,45977,31057,25032,15388],
    textposition = "auto",
    marker = {"color":  ['rgb(222,235,247)', 'rgb(198,219,239)', 'rgb(158,202,225)', 'rgb(107,174,214)', 'rgb(39,116,174)', 'rgb(0,85,135)', 'rgb(0,59,92)']},
    textinfo = "value+percent previous",textfont_size=13))

fig.add_trace(go.Funnel(
    name = 'Gsearch-custom_lander',
    y = ["sessions", "to_products", "to_mrfuzzy", "to_cart","to_shipping", "to_billing", "to_thankyou"],
    x = [13005,5128,3660,1592,1067,881,381],
    marker = {"color":  ['rgb(255,247,188)', 'rgb(254,227,145)', 'rgb(255,199,44)', 'rgb(255,184,28)', 'rgb(255,184,28)', 'rgb(255,184,28)', 'rgb(255,184,28)']},
    textinfo = "value+percent previous",textfont_size=13))

fig.update_layout( height=650, width=1000, 
                  title_text="Conversion Funnel for homepage and custom lander",
                 font_size=15)
fig.show()


# In[20]:


# Visualization: conversion funnels 
# see https://plotly.com/python/funnel-charts/

from plotly import graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Funnelarea(
    text = ["sessions", "to_products", "to_mrfuzzy", "to_cart","to_shipping", "to_billing", "to_thankyou"],
    values = [234559,129391, 102842, 46061, 31119, 25080, 15416],
    marker = {"colors": ['rgb(222,235,247)', 'rgb(198,219,239)', 'rgb(158,202,225)', 'rgb(107,174,214)', 'rgb(66,146,198)', 'rgb(33,113,181)', 'rgb(8,81,156)']},
    title = {"position": "top center", "text": "Custom Lander Segment"},
    domain = {"x": [0, 0.5], "y": [1, 1]}))

fig.add_trace(
   go.Funnelarea(
    text = ["sessions", "to_products", "to_mrfuzzy", "to_cart","to_shipping", "to_billing", "to_thankyou"],
    values = [13005, 5128, 3660, 1592, 1067, 881, 381],
    title = {"position": "top center", "text": "Homepage Segment"},
    domain = {"x": [0.5, 1], "y": [1, 1]}))
              
fig.update_layout( height=500, width=900, title_text="Conversion Funnel for 2 Segments of Gsearch")
fig.show()


# In[ ]:




