import pymysql
import traceback

geonamesDB = pymysql.connect(host="host", user="user", password="password", db="geonames_db", port=3306)
bdbkDB = pymysql.connect(host="host", user="user", password="password", db="bdbk_db", port=3306)
TABLE_GEO_NAMES = "geoname_merge_zhjian"
TABLE_TITLE = 'title'
TABLE_SUMMARY = 'summary'

def getAllGeoNames():
    cursor = geonamesDB.cursor()
    geonames={}
    sql= 'select * from '+TABLE_GEO_NAMES
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for result in results:
            name = result[1]
            names=name.split('|')
            for n in names:
                if n not in geonames:
                    geonames[n]=[(result[0], n, result[2], result[3], result[4], result[5], result[6], result[7],
                                     result[8], result[9], result[10])]
                else:
                    geonames[n].append((result[0],n,result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10]))
            # geonames.append(result)
        # geonames=results
    except Exception as e:
        print('error', e)
        traceback.print_exc()
    finally:
        cursor.close()
        return geonames

def getUrlSummaryByname(name):
    cursor = bdbkDB.cursor()
    urlSummarys=[]
    sql = 'select url from ' + TABLE_TITLE+' where title = \''+name+'\''
    try:
        # print(sql)
        cursor.execute(sql)
        results = cursor.fetchall()
        for result in results:
            url = result[0]
            sql = 'select summary from ' + TABLE_SUMMARY + ' where url =\'' + url+'\''
            cursor.execute(sql)
            results = cursor.fetchall()
            if results:
                urlSummarys.append((url,name,results[0][0]))
    except Exception as e:
        print('error', e)
        traceback.print_exc()
    finally:
        cursor.close()
        return urlSummarys
