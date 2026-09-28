"""Location/channel metrics calculated from the selected source records."""
from functools import lru_cache
from pathlib import Path
import pandas as pd
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[2] / 'processed_data' / 'cleaned'


@lru_cache(maxsize=24)
def _read_cached(path, modified):
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)


def read(name):
    path = ROOT / name / f'{name}.parquet'
    if not path.exists():
        path = path.with_suffix('.csv')
    if not path.exists():
        raise HTTPException(503, f'{name} dataset is unavailable. Run the data pipeline first.')
    return _read_cached(path, path.stat().st_mtime_ns).copy()


def dates(frame, column, fallback=None):
    result = pd.to_datetime(frame[column], errors='coerce')
    if fallback and fallback in frame:
        result = result.fillna(pd.to_datetime(frame[fallback], errors='coerce'))
    return result.dt.normalize()


def scope(date_range='ALL', location_ids=None, city=None, tier=None):
    orders = read('orders')
    orders = orders[orders.order_status.eq('COMPLETED')].copy()
    orders['_date'] = dates(orders, 'order_date', 'order_timestamp')
    end = orders['_date'].max()
    dr = (date_range or 'ALL').strip().upper()
    if dr == 'ALL':
        start = orders['_date'].min()
    elif dr in ('30_DAYS', 'LAST_30_DAYS', '90_DAYS', 'LAST_90_DAYS'):
        start = end - pd.Timedelta(days=29 if '30' in dr else 89)
    elif dr in ('Q1', 'Q2', 'Q3', 'Q4'):
        start = pd.Timestamp(end.year, 1 + (int(dr[1])-1)*3, 1)
        end = start + pd.offsets.QuarterEnd()
    else:
        raise HTTPException(422, 'Unsupported date range')
    stores = read('restaurants')
    if location_ids and location_ids.upper() != 'ALL':
        ids = [v.strip().upper() for v in location_ids.split(',') if v.strip()]
        stores = stores[stores.location_id.isin(ids)]
    if city and city.upper() != 'ALL':
        stores = stores[stores.city.str.casefold().eq(city.strip().casefold())]
    if tier and tier.upper() != 'ALL':
        stores = stores[stores.location_tier.str.casefold().eq(tier.strip().casefold())]
    orders = orders[orders['_date'].between(start,end) & orders.location_id.isin(stores.location_id)].copy()
    lines = read('order_items')
    lines = lines[lines.order_id.isin(orders.order_id)]
    menu = read('menu_items')
    lines = lines.merge(menu[['item_id','cost_price']], on='item_id', how='left', validate='many_to_one')
    lines['_cost'] = lines.quantity * lines.cost_price
    costs = lines.groupby('order_id')['_cost'].sum(min_count=1)
    orders['_cost'] = orders.order_id.map(costs)
    orders['_units'] = orders.order_id.map(lines.groupby('order_id').quantity.sum())
    orders['_items'] = orders.order_id.map(lines.groupby('order_id').item_id.nunique())
    orders['_revenue'] = orders.subtotal_amount - orders.discount_amount
    orders['_profit'] = orders['_revenue'] - orders['_cost']
    return orders, lines, stores, start, end


def number(value, digits=2):
    return None if pd.isna(value) else round(float(value),digits)


def ratio(a,b):
    return round(float(a/b*100),1) if b else 0.0


def metadata(orders,start,end):
    return {'start_date':str(start.date()),'end_date':str(end.date()),
            'date_anchor':'Latest completed order in the dataset',
            'revenue_definition':'Completed order subtotal less discount; excludes tax, tips and delivery fees.',
            'profit_definition':'Estimated using current menu costs. Orders without retained item costs are excluded from profit.',
            'orders_missing_costs':int(orders['_cost'].isna().sum())}


def performance(date_range='ALL', location_ids=None, city=None, tier=None):
    orders,lines,stores,start,end = scope(date_range,location_ids,city,tier)
    waste = read('wastage')
    waste = waste[dates(waste,'wastage_date').between(start,end) & waste.location_id.isin(stores.location_id)]
    ratings = read('ratings')
    ratings = ratings[dates(ratings,'review_date').between(start,end) & ratings.location_id.isin(stores.location_id)]
    result=[]
    for _,store in stores.iterrows():
        o=orders[orders.location_id.eq(store.location_id)]
        w=waste[waste.location_id.eq(store.location_id)]
        r=ratings[ratings.location_id.eq(store.location_id)]
        revenue=float(o['_revenue'].sum()); profit=float(o['_profit'].sum())
        known_revenue=float(o.loc[o['_cost'].notna(),'_revenue'].sum())
        units=float(o['_units'].sum()); wasted=float(w.quantity_wasted.sum())
        customers=o.loc[o.customer_id.ne('CUST-GUEST'),'customer_id'].value_counts()
        waste_pct=ratio(wasted,units+wasted)
        result.append({'location_id':store.location_id,'restaurant_name':store['name'],
            'city':store.city,'state':store.state,'location_tier':store.location_tier,
            'revenue':number(revenue),'profit':number(profit),'margin_pct':ratio(profit,known_revenue),
            'orders':len(o),'aov':number(revenue/len(o)) if len(o) else 0,
            'customers':len(customers),'repeat_customers_pct':ratio(int((customers>1).sum()),len(customers)),
            'avg_overall_rating':number(r.overall_rating.mean()),'avg_food_rating':number(r.food_rating.mean()),
            'avg_service_rating':number(r.service_rating.mean()),'avg_ambiance_rating':number(r.ambiance_rating.mean()),
            'rating_count':len(r),'csat_pct':ratio(int((r.overall_rating>=4).sum()),len(r)),
            'quantity_wasted':int(wasted),'wastage_loss':number(w.total_loss_amount.sum()),
            'wastage_rate_pct':waste_pct,'wastage_risk':'High' if waste_pct>=15.5 else 'Medium' if waste_pct>=14 else 'Low'})
    result.sort(key=lambda x:x['revenue'],reverse=True)
    for i,row in enumerate(result): row['revenue_rank']=i+1
    for i,row in enumerate(sorted(result,key=lambda x:x['profit'],reverse=True)): row['profit_rank']=i+1
    revenue=float(orders['_revenue'].sum()); profit=float(orders['_profit'].sum())
    kpis={'total_locations':len(stores),'total_revenue':number(revenue),'total_profit':number(profit),
        'total_orders':len(orders),'average_order_value':number(revenue/len(orders)) if len(orders) else 0,
        'active_customers':int(orders.loc[orders.customer_id.ne('CUST-GUEST'),'customer_id'].nunique()),
        'average_rating':number(ratings.overall_rating.mean()),'wastage_cost':number(waste.total_loss_amount.sum()),
        'avg_wastage_pct':ratio(waste.quantity_wasted.sum(),orders['_units'].sum()+waste.quantity_wasted.sum()),
        'contribution_margin_pct':ratio(profit,orders.loc[orders['_cost'].notna(),'_revenue'].sum())}
    all_stores=read('restaurants')
    return {'kpis':kpis,'locations':result,'cities':sorted(all_stores.city.dropna().unique().tolist()),
        'tiers':sorted(all_stores.location_tier.dropna().unique().tolist()),'scope':metadata(orders,start,end),
        'revenue_profit_chart':[dict(r,name=r['restaurant_name']) for r in result],
        'orders_aov_chart':[dict(r,name=r['restaurant_name']) for r in result],
        'wastage_rankings':[dict(r,name=r['restaurant_name'],wastage_cost=r['wastage_loss']) for r in sorted(result,key=lambda r:r['wastage_loss'],reverse=True)],
        'ratings_rankings':sorted(result,key=lambda r:r['avg_overall_rating'] or 0,reverse=True),'insights':[]}


def channels(date_range='ALL', location_id=None):
    orders,lines,stores,start,end=scope(date_range,location_id)
    labels={'DINE_IN':'Dine-in','TAKEOUT':'Takeaway','DELIVERY':'Delivery','DRIVE_THRU':'Drive-thru'}
    orders['_channel']=orders.order_type.map(labels).fillna(orders.order_type)
    orders['_hour']=pd.to_datetime(orders.order_time,format='%H:%M:%S',errors='coerce').dt.hour
    rows=[]; hourly=[]
    revenue=float(orders['_revenue'].sum()); profit=float(orders['_profit'].sum())
    for name,g in orders.groupby('_channel'):
        rev=float(g['_revenue'].sum()); gp=float(g['_profit'].sum())
        counts=g['_hour'].dropna().astype(int).value_counts()
        days=g['_date'].dt.day_name().value_counts()
        promo=g.promotion_id.fillna('').ne(''); discount=g.discount_amount.gt(0)
        rows.append({'channel':name,'orders':len(g),'order_share_pct':ratio(len(g),len(orders)),
            'revenue':number(rev),'revenue_share_pct':ratio(rev,revenue),'profit':number(gp),'profit_share_pct':ratio(gp,profit),
            'cost':number(g['_cost'].sum()),'margin_pct':ratio(gp,g.loc[g['_cost'].notna(),'_revenue'].sum()),
            'aov':number(rev/len(g)),'avg_units_per_order':number(g['_units'].mean()),
            'avg_distinct_items_per_order':number(g['_items'].mean()),'discount_penetration_pct':ratio(discount.sum(),len(g)),
            'avg_discount_when_applied':number(g.loc[discount,'discount_amount'].mean()),
            'promo_order_penetration_pct':ratio(promo.sum(),len(g)),
            'promoted_revenue_share_pct':ratio(g.loc[promo,'_revenue'].sum(),rev),
            'peak_ordering_hour':f'{int(counts.idxmax()):02}:00' if len(counts) else 'Unavailable',
            'peak_day_of_week':days.idxmax() if len(days) else 'Unavailable',
            'weekend_share_pct':ratio(g['_date'].dt.dayofweek.isin([5,6]).sum(),len(g)),
            'lunch_peak_share_pct':ratio(g['_hour'].between(12,14).sum(),len(g)),
            'dinner_peak_share_pct':ratio(g['_hour'].between(18,21).sum(),len(g))})
        hourly.append({'channel':name,'distribution':{str(h):int(counts.get(h,0)) for h in range(24)}})
    rows.sort(key=lambda r:r['revenue'],reverse=True)
    kpis={'total_channel_orders':len(orders),'total_channel_revenue':number(revenue),'total_channel_profit':number(profit)}
    for name,key in [('top_orders_channel','orders'),('top_revenue_channel','revenue'),('top_profit_channel','profit'),('highest_aov_channel','aov'),('largest_basket_channel','avg_units_per_order')]:
        kpis[name]=max(rows,key=lambda r:r[key] or 0) if rows else None
    details=lines.merge(orders[['order_id','_channel']],on='order_id',validate='many_to_one')
    details=details.merge(read('menu_items')[['item_id','category_id']],on='item_id',validate='many_to_one')
    details=details.merge(read('menu_categories')[['category_id','category_name']],on='category_id',validate='many_to_one')
    details['_line_profit']=details.item_total-details['_cost']
    preferences=[]
    for (channel,category),g in details.groupby(['_channel','category_name']):
        total_units=details.loc[details['_channel'].eq(channel),'quantity'].sum()
        preferences.append({'channel':channel,'category_name':category,'units_sold':int(g.quantity.sum()),
            'category_revenue':number(g.item_total.sum()),'category_gross_profit':number(g['_line_profit'].sum()),
            'channel_unit_share_pct':ratio(g.quantity.sum(),total_units)})
    return {'kpis':kpis,'channel_performance':rows,'hourly_patterns':hourly,'menu_preferences':preferences,
        'location_channel_matrix':pd.crosstab(orders.location_id,orders['_channel']).reset_index().to_dict('records'),
        'insights':[],'scope':metadata(orders,start,end)}
