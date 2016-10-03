# use search;
# select pinid, view, query from daily_pin_perf
# where dt="2016-09-29" and query is not NULL and view="search_pins";
#
# select id from db_pins_snapshots where dt="2016-09-29";

"""
select pinid, view, query, json
from (
  select pinid, view, query from search.daily_pin_perf
  where dt="2016-09-29" and query is not NULL and view="search_pins") a
join (
  select id, json from db_pins_snapshots where dt="2016-09-29") b
on (a.pinid = b.id);
"""