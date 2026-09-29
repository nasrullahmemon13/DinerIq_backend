"""Generate deterministic synthetic restaurant data; no real customer records."""
from datetime import date, timedelta
import random


def demo_rows():
    rng = random.Random(42)
    data = {name: [] for name in ['restaurants','menu_categories','menu_items','customers','promotions','pricing_history','orders','order_items','ratings','wastage','inventory']}
    for n in range(1,6):
        data['restaurants'].append(dict(restaurant_id=f'DEMO-R{n}',location_id=f'DEMO-L{n}',name=f'Demo Restaurant {n}',city=['Karachi','Lahore','Islamabad','Hyderabad','Multan'][n-1],state='PK',country='Pakistan',operating_status='ACTIVE',seating_capacity=80,cost_index=1.0))
    for n,name in enumerate(['Burgers','Pizza','Rice','Drinks'],1):
        data['menu_categories'].append(dict(category_id=f'DEMO-CAT{n}',name=name,description='Synthetic demo category',target_margin_pct=65,is_active=True))
    for n in range(1,21):
        price=round(6+n*.7,2)
        data['menu_items'].append(dict(item_id=f'DEMO-I{n}',category_id=f'DEMO-CAT{(n-1)//5+1}',name=f'Demo {data["menu_categories"][(n-1)//5]["name"]} {n}',base_price=price,cost_price=round(price*.35,2),margin_pct=65,is_active=True,prep_time_minutes=15,shelf_life_days=3))
    for n in range(1,101):
        data['customers'].append(dict(customer_id=f'DEMO-C{n}',first_name='Demo',last_name=f'Customer {n}',email=f'customer{n}@example.invalid',customer_segment=['Champions','Loyal','New'][n%3],loyalty_tier=['Gold','Silver','Bronze'][n%3],loyalty_points=n*10,signup_date=date(2025,1,1),preferred_location_id=f'DEMO-L{n%5+1}',is_active=True,churn_risk_score=.2))
    for n in range(1,4):
        data['promotions'].append(dict(promotion_id=f'DEMO-P{n}',promotion_name=f'Demo Offer {n}',discount_type='Percentage',discount_value=10,min_order_amount=10,start_date=date(2025,1,1),end_date=date(2025,12,31),description='Synthetic demonstration offer'))
    for n in range(1,1201):
        day=date(2025,1,1)+timedelta(days=(n-1)%365)
        location=f'DEMO-L{n%5+1}'; customer=f'DEMO-C{n%100+1}'; order=f'DEMO-O{n}'
        subtotal=0
        for j in range(2):
            item=rng.choice(data['menu_items']); qty=rng.randint(1,3); amount=round(qty*item['base_price'],2); subtotal+=amount
            data['order_items'].append(dict(order_item_id=f'DEMO-OI{n}-{j}',order_id=order,item_id=item['item_id'],quantity=qty,unit_price=item['base_price'],subtotal=amount,item_discount=0,item_total=amount))
        subtotal=round(subtotal,2); tax=round(subtotal*.05,2)
        data['orders'].append(dict(order_id=order,customer_id=customer,location_id=location,order_date=day,order_time='13:00:00',order_timestamp=day.isoformat()+'T13:00:00',order_type=['Dine-In','Takeaway','Delivery'][n%3],order_status='Completed',payment_method='Card',subtotal_amount=subtotal,discount_amount=0,tax_amount=tax,tip_amount=0,delivery_fee=0,total_amount=round(subtotal+tax,2)))
        if n%4==0:
            data['ratings'].append(dict(rating_id=f'DEMO-RT{n}',order_id=order,customer_id=customer,item_id=item['item_id'],location_id=location,overall_rating=rng.randint(3,5),food_rating=4,service_rating=4,ambiance_rating=4,review_text='Synthetic demo review',review_date=day))
    for n in range(1,241):
        item=data['menu_items'][n%20]; loc=f'DEMO-L{n%5+1}'; day=date(2025,1,1)+timedelta(days=n%365); quantity=n%3+1
        data['wastage'].append(dict(wastage_id=f'DEMO-W{n}',item_id=item['item_id'],location_id=loc,wastage_date=day,quantity_wasted=quantity,unit_cost=item['cost_price'],total_loss_amount=round(quantity*item['cost_price'],2),wastage_reason='Overproduction',reported_by='Demo generator'))
    for loc in data['restaurants']:
        for item in data['menu_items']:
            key=loc['location_id']+'-'+item['item_id']
            data['inventory'].append(dict(inventory_id=key,location_id=loc['location_id'],item_id=item['item_id'],snapshot_date=date(2025,12,31),starting_stock=100,quantity_received=20,quantity_sold=40,quantity_wasted=2,ending_stock=78,reorder_point=20,stock_status='Healthy'))
            data['pricing_history'].append(dict(price_history_id=key,location_id=loc['location_id'],item_id=item['item_id'],base_price=item['base_price'],cost_price=item['cost_price'],effective_start_date=date(2025,1,1),change_reason='Synthetic demo initial price'))
    return data


def write_sql(path):
    from pathlib import Path
    from sqlalchemy.dialects.postgresql import insert, dialect
    from database.models import Base
    rows=demo_rows()
    statements=['DO $dineiq_demo$ BEGIN']
    for name, records in rows.items():
        stmt=insert(Base.metadata.tables[name]).values(records).on_conflict_do_nothing()
        statements.append(str(stmt.compile(dialect=dialect(),compile_kwargs={'literal_binds':True}))+';')
    statements.append('END $dineiq_demo$;')
    Path(path).write_text('\n'.join(statements),encoding='utf-8')
    return {name:len(records) for name,records in rows.items()}


if __name__ == '__main__':
    import sys
    print(write_sql(sys.argv[1]))
