libname p clear;

libname p 'F:\Engagements\ENG9\Patrick\Delta';
options compress = yes fullstimer errors = 2 obs = max;
ods html newfile=proc;

data p.delta_rfp;
infile 'F:\Engagements\ENG9\Patrick\Delta\sample_dataset_rfp_20152704.csv' delimiter = ',' dsd pad missover lrecl = 32767 firstobs = 2;
informat
response_dt date9.
PFL_FLEET_TYP_CD $10.00 
gts_1 best32.
gts_12 best32.
gts_13 best32.
gts_14 best32.
gts_15 best32.
comped_med best32.
natural_med best32.
never_comped best32.
lapsed_med best32.
Lesiure_tkts_yr1 best32.
Business_tkts_yr1 best32.
total_ap_yr1 best32.
int_tkts_yr1 best32.
dom_tkts_yr1 best32.
dl_miles_flown_yr1 best32.
dl_segs_yr1 best32.
dl_seg_rev_yr1 best32.
F_tkts_yr1 best32.
F_tkt_rev_yr1 best32.
F_miles_flown_yr1 best32.
C_tkts_yr1 best32.
C_tkt_rev_yr1 best32.
C_miles_flown_yr1 best32.
Y_tkts_yr1 best32.
Y_tkt_rev_yr1 best32.
Y_miles_flown_yr1 best32.
F_upsell_tkts_yr1 best32.
F_upsell_tkt_rev_yr1 best32.
F_upsell_miles_flown_yr1 best32.
total_miles_flown_yr1 best32.
total_tkts_yr1 best32.
total_tkt_rev_yr1 best32.
dist_partners best32.
tenure best32.
DSC_Entries	best32.
hhi	best32.;
format
response_dt	mmddyy10.
PFL_FLEET_TYP_CD $10.
gts_1 8.
gts_12 8.
gts_13 8.
gts_14 8.
gts_15 8.
comped_med	8.
natural_med	8.
never_comped 8.
lapsed_med 8.
Lesiure_tkts_yr1 8.
Business_tkts_yr1 8.
total_ap_yr1 8.
int_tkts_yr1 8.
dom_tkts_yr1 8.
dl_miles_flown_yr1 8.
dl_segs_yr1 8.
dl_seg_rev_yr1 8.
F_tkts_yr1 8.
F_tkt_rev_yr1 8.
F_miles_flown_yr1 8.
C_tkts_yr1 8.
C_tkt_rev_yr1 8.
C_miles_flown_yr1 8.
Y_tkts_yr1 8.
Y_tkt_rev_yr1 8.
Y_miles_flown_yr1 8.
F_upsell_tkts_yr1 8.
F_upsell_tkt_rev_yr1 8.
F_upsell_miles_flown_yr1 8.
total_miles_flown_yr1 8.
total_tkts_yr1 8.
total_tkt_rev_yr1 8.
dist_partners 8.
tenure 8.
DSC_Entries 8.
hhi 8.;
input
response_dt
PFL_FLEET_TYP_CD $
gts_1
gts_12
gts_13
gts_14
gts_15
comped_med
natural_med
never_comped
lapsed_med
Lesiure_tkts_yr1
Business_tkts_yr1
total_ap_yr1
int_tkts_yr1
dom_tkts_yr1
dl_miles_flown_yr1
dl_segs_yr1
dl_seg_rev_yr1
F_tkts_yr1
F_tkt_rev_yr1
F_miles_flown_yr1
C_tkts_yr1
C_tkt_rev_yr1
C_miles_flown_yr1
Y_tkts_yr1
Y_tkt_rev_yr1
Y_miles_flown_yr1
F_upsell_tkts_yr1
F_upsell_tkt_rev_yr1
F_upsell_miles_flown_yr1
total_miles_flown_yr1
total_tkts_yr1
total_tkt_rev_yr1
dist_partners
tenure
DSC_Entries
hhi;
run;

proc freq data = p.delta_rfp;
tables gts_1 gts_12 gts_13 gts_14 gts_15 response_dt PFL_FLEET_TYP_CD/missing;
run;

proc means data = p.delta_rfp min p1 p5 p10 p25 p50 p75 p90 p95 p99 max;
var total_tkt_rev_yr1;
run;

/*Odds Testing.  Not really sure if this even works because depvar could be very far in the past whereas IV's may be for the full 2014 calendar year*/
data test;
set p.delta_rfp;
if (25000<=total_miles_flown_yr1<50000 or 30<=dl_segs_yr1<60) and total_tkt_rev_yr1 >= 3000 then medallion_status = 'Silver    ';
if (50000<=total_miles_flown_yr1<75000 or 60<=dl_segs_yr1<100) and total_tkt_rev_yr1 >= 6000 then medallion_status = 'Gold';
if (75000<=total_miles_flown_yr1<125000 or 100<=dl_segs_yr1<140) and total_tkt_rev_yr1 >= 9000 then medallion_status = 'Platinum';
if (125000<=total_miles_flown_yr1 or 140<=dl_segs_yr1) and total_tkt_rev_yr1 >= 15000 then medallion_status = 'Diamond';
if 25000<=total_miles_flown_yr1<50000 and total_tkt_rev_yr1 >= 3000 then med_status = 'Miles Silver      ';
if 30<=dl_segs_yr1<60 and total_tkt_rev_yr1 >= 3000 then med_status = 'Segment Silver    ';
if 50000<=total_miles_flown_yr1<75000 and total_tkt_rev_yr1 >= 6000 then med_status = 'Miles Gold';
if 60<=dl_segs_yr1<100 and total_tkt_rev_yr1 >= 6000 then med_status = 'Segment Gold';
if 75000<=total_miles_flown_yr1<125000 and total_tkt_rev_yr1 >= 9000 then med_status = 'Miles Platinum';
if 100<=dl_segs_yr1<140 and total_tkt_rev_yr1 >= 9000 then med_status = 'Segment Platinum';
if 125000<=total_miles_flown_yr1 and total_tkt_rev_yr1 >= 15000 then med_status = 'Miles Diamond';
if 140<=dl_segs_yr1 and total_tkt_rev_yr1 >= 15000 then med_status = 'Segment Diamond';
id = _N_;
if gts_1 = 1 then depvar = 1; else if gts_1 = 5 then depvar = 0;
if depvar = 1 then eqwt = 22476/1247; else if depvar = 0 then eqwt = 1;
if total_tkt_rev_yr1 < 1000 then depvar1 = 0;
else if total_tkt_rev_yr1 > 10000 then depvar1 = 1;
if depvar1 = 1 then eqwt1 = 22015/5956; else if depvar1 = 0 then eqwt1 = 1;
if gts_12 = 1 then depvar2 = 1; else if gts_12 = 5 then depvar2 = 0;
if depvar2 = 1 then eqwt2 = 25146/761; else if depvar2 = 0 then eqwt2 = 1;
if gts_13 = 1 then depvar3 = 1; else if gts_13 = 5 then depvar3 = 0;
if depvar3 = 1 then eqwt3 = 17405/313; else if depvar3 = 0 then eqwt3 = 1;
if response_dt > mdy(12,31,2014) then do;
if gts_1 in (1,2,3) then depvar4 = 1; else if gts_1 = 5 then depvar4 = 0;
if depvar4 = 1 then eqwt4 = 10000/107; else if depvar4 = 0 then eqwt4 = 10000/235;
end;
lei_bus_ratio = Lesiure_tkts_yr1/Business_tkts_yr1;
dom_int_ratio = dom_tkts_yr1/int_tkts_yr1;
pct_trav_leis = Lesiure_tkts_yr1/total_tkts_yr1;
avg_rev_per_tkt = total_tkt_rev_yr1/total_tkts_yr1;
dl_rev_per_mile = dl_seg_rev_yr1/dl_miles_flown_yr1;
f_rev_per_mile = F_tkt_rev_yr1/F_miles_flown_yr1;
f_miles_per_tkt = F_miles_flown_yr1/F_tkts_yr1;
c_rev_per_mile = c_tkt_rev_yr1/c_miles_flown_yr1;
c_miles_per_tkt = c_miles_flown_yr1/c_tkts_yr1;
y_rev_per_mile = Y_tkt_rev_yr1/Y_miles_flown_yr1;
y_miles_per_tkt = Y_miles_flown_yr1/Y_tkts_yr1;
f_upsell_rev_per_mile = F_upsell_tkt_rev_yr1/F_upsell_miles_flown_yr1;
f_upsell_miles_per_tkt = F_upsell_miles_flown_yr1/F_upsell_tkts_yr1;
pct_tkts_f = F_tkts_yr1/total_tkts_yr1;
pct_tkts_f_upsold = F_upsell_tkts_yr1/F_tkts_yr1;
pct_tkts_c = c_tkts_yr1/total_tkts_yr1;
pct_tkts_y = y_tkts_yr1/total_tkts_yr1;
pct_miles_flown_f = F_miles_flown_yr1/total_miles_flown_yr1;
pct_miles_f_upsold = F_upsell_miles_flown_yr1/F_miles_flown_yr1;
pct_miles_flown_c = c_miles_flown_yr1/total_miles_flown_yr1;
pct_miles_flown_y = y_miles_flown_yr1/total_miles_flown_yr1;
pct_tkt_rev_f = F_tkt_rev_yr1/total_tkt_rev_yr1;
pct_tkt_rev_f_upsold = F_upsell_tkt_rev_yr1/F_tkt_rev_yr1;
pct_tkt_rev_c = c_tkt_rev_yr1/total_tkt_rev_yr1;
pct_tkt_rev_y = y_tkt_rev_yr1/total_tkt_rev_yr1;
pct_dl_ttl_miles = dl_miles_flown_yr1/total_miles_flown_yr1;
pct_dl_ttl_rev = dl_seg_rev_yr1/total_tkt_rev_yr1;
dsc_entry_per_trip = total_tkts_yr1/DSC_Entries;
rto_Fmiles_Cmiles = F_miles_flown_yr1/C_miles_flown_yr1;
rto_Fmiles_Ymiles = F_miles_flown_yr1/Y_miles_flown_yr1;
rto_Fmiles_F_upsell_miles = F_miles_flown_yr1/F_upsell_miles_flown_yr1;
rto_Cmiles_Ymiles = C_miles_flown_yr1/Y_miles_flown_yr1;
rto_Cmiles_F_upsell_miles = C_miles_flown_yr1/F_upsell_miles_flown_yr1;
rto_Ymiles_F_upsell_miles = Y_miles_flown_yr1/F_upsell_miles_flown_yr1;
run;

proc contents data = test varnum; run;

proc freq data = test; tables medallion_status med_status/missing; run;

proc freq data = test; tables depvar/missing; weight eqwt; run;
proc freq data = test; tables depvar1/missing; weight eqwt1; run;
proc freq data = test; tables depvar2/missing; weight eqwt2; run;
proc freq data = test; tables depvar3/missing; weight eqwt3; run;
proc freq data = test; tables depvar4/missing; *weight eqwt4; run;
proc freq data = test; tables pfl_fleet_typ_cd/missing; run;
proc freq data = test; tables comped_med natural_med never_comped lapsed_med/missing; run;

proc means data = test min p1 p5 p10 p25 p50 p75 p90 p95 p99 max mean stddev var;
var tenure;
run;

/*Clustering Testing.  Let's see if we can break anything out of this to show seperation between travelers*/
%include 'F:\Engagements\ENG9\Patrick\Delta\Clustering Varlist.sas';

proc means data = test mean;
var %varlist;
run;

%include 'F:\Engagements\ENG9\Patrick\Delta\Clustering Recode.sas';

data test2;
set test;
%recode;
run;

proc standard data = test2 out = test3 mean=0 std=1; var %varlist; run; quit;

proc cluster data = test3 simple noeigen method = ward rmsstd rsquare nonorm out = tree /*noprint*/;
id id;
var %varlist;
run;

DATA T2;
	SET TREE;
	WHERE _NAME_ CONTAINS 'CL';
DATA T3;
	SET T2;
	KEEP _NCL_ _RMSSTD_--_RSQ_;
PROC SORT DATA = T3 OUT = T4;
	BY _NCL_;
RUN;

proc print data = t4; run;

PROC TREE DATA = TREE OUT = CLUS1 NCLUSTER = 5 NOPRINT;
	ID id;
	COPY %varlist;
*OBTAINING CENTROID MEANS;
PROC MEANS DATA = CLUS1;
	CLASS CLUSTER;
	OUTPUT OUT = INITIAL MEAN = %varlist;
	VAR %varlist;
RUN;

*NONHIERARCHICAL CLUSTERING;
PROC FASTCLUS DATA = test3 SEED = INITIAL DISTANCE RADIUS = 0 REPLACE = PART MAXCLUSTERS = 5 NOPRINT OUT = FINAL 
MEAN = FINAL_MEANS  outstat = final_stat;
	VAR %varlist;
RUN;

proc print data = final_stat;
var _TYPE_;
run;

proc transpose data = final_stat out = rsquare(rename = (COL1 = INITIAL1 COL2 = INITIAL2 COL3 = INITIAL3 COL4 = INITIAL4 COL5 = INITIAL5 COL6 = LEAST COL7 = CRITERION
COL8 = MEAN COL9 = STD COL10 = WITHIN_STD COL11 = RSQ COL12 = RSQ_RATIO COL13 = PSEUDO_F COL14 = ERSQ COL15 = CCC COL16 = SEED1 COL17 = SEED2 COL18 = SEED3 COL19 = SEED4
COL20 = SEED5 COL21 = CENTER1 COL22 = CENTER2 COL23 = CENTER3 COL24 = CENTER4 COL25 = CENTER5 COL26 = DISPERSION1 COL27 = DISPERSION2 COL28 = DISPERSION3 COL29 = DISPERSION4
COL30 = DISPERSION5 COL31 = FREQ1 COL32 = FREQ2 COL33 = FREQ3 COL34 = FREQ4 COL35 = FREQ5));
proc sort; by descending rsq; run;

data _null_;
array one %varlist;
dim_one = dim(one);
call symput('it',dim_one);
run;

%put &it;

data _null_;
set final_stat(where = (_type_ = 'SEED'))end = last;
if last then call symput('nclust',cluster);
run;
%put &nclust;

options mprint symbolgen mlogic;

%macro centro;
filename clus 'F:\Engagements\ENG9\Patrick\Delta\clust test dist.sas';
data _null_;
set final_stat(where = (_type_ = 'SEED')) end = last;
length sm1 $50;
file clus;
if _n_ = 1 then put '%macro distancex;';
dst = catt("distance",cluster,"=sum(0");
put dst;
%do j = 1 %to &it;
sm0 = ',(';
sm1 = "%scan(%varlist,&j,' ')";
sm2 = %scan(%varlist,&j,' ');
sm3 = ')**2';
sm = catt(sm0,sm1,'-',sm2,sm3);
put @ 015 sm;
%end;
nodst = catt(')**(1/2);');
put @ 015 nodst;
if last then put '%mend distancex;';
run;
%mend centro;
%centro;

%include 'F:\Engagements\ENG9\Patrick\Delta\clust test dist.sas';

%macro assignclust;
data output_cluster;
set test3;
%distancex;
my_min = min(%do i = 1 %to &nclust;
			   %if &i = &nclust %then %do;
			     distance&i%end;
			   %else %do;
			     distance&i,%end;
			  %end;);
length xclust $3;
array dist %do i = 1 %to &nclust;
distance&i
%end;;
do i = 1 to &nclust;
if my_min = dist(i) then do;
%let imhere = i;
xclust = &imhere;
i+5;
end;
end;
run;
%mend assignclust;
%assignclust;

proc freq data = output_cluster; tables xclust/missing; run;

proc sql;
create table final_clusters as
select
a.*,
b.cluster
from test2 a left join final b
on a.id = b.id;

create table final_clusters1 as
select
a.*,
b.xclust
from final_clusters a left join output_cluster b
on a.id = b.id;

create table p.final_clusters2 as
select
a.*,
b.medallion_status,
b.med_status
from final_clusters1 a left join test b
on a.id = b.id;
quit;

%macro m;
%do k = 1 %to &it;
%global var&k.;
%let mo&k. = %scan(%varlist,&k.,' ');
%put &&mo&k.;
data _null_;
newone = kcompress("&&mo&k.",'');
call symput("var&k.",trim(newone));
run;
%put var&k.;
%end;
%mend m;
%m;
%put &var1 &var27;

%macro desc;
proc sql;
create table descriptions as
select
cluster,
%do i = 1 %to &it.;
	avg(&&var&i..) as &&var&i..,
%end;
sum(case when medallion_status = 'Silver' then 1 else 0 end) as num_silver,
sum(case when medallion_status = 'Gold' then 1 else 0 end) as num_gold,
sum(case when medallion_status = 'Platinum' then 1 else 0 end) as num_platinum,
sum(case when medallion_status = 'Diamond' then 1 else 0 end) as num_diamond,
sum(case when med_status = 'Miles Silver' then 1 else 0 end) as num_miles_silver,
sum(case when med_status = 'Segment Silver' then 1 else 0 end) as num_seg_silver,
sum(case when med_status = 'Miles Gold' then 1 else 0 end) as num_miles_gold,
sum(case when med_status = 'Segment Gold' then 1 else 0 end) as num_seg_gold,
sum(case when med_status = 'Miles Platinum' then 1 else 0 end) as num_miles_platinum,
sum(case when med_status = 'Segment Platinum' then 1 else 0 end) as num_seg_platinum,
sum(case when med_status = 'Miles Diamond' then 1 else 0 end) as num_miles_diamond,
sum(case when med_status = 'Segment Diamond' then 1 else 0 end) as num_seg_diamond,
sum(case when 0<=tenure<6 then 1 else 0 end) as tenure_0_5,
sum(case when 6<=tenure<15 then 1 else 0 end) as tenure_6_14,
sum(case when 15<=tenure<23 then 1 else 0 end) as tenure_15_22,
sum(case when tenure>=23 then 1 else 0 end) as tenure_GT_22,
sum(case when comped_med = 1 then 1 else 0 end) as num_comped_med,
sum(case when natural_med = 1 then 1 else 0 end) as num_natural_med,
sum(case when never_comped = 1 then 1 else 0 end) as num_never_comped,
sum(case when lapsed_med = 1 then 1 else 0 end) as num_lapsed_med,
sum(case when PFL_FLEET_TYP_CD = '319' then 1 else 0 end) as aircraft_319,
sum(case when PFL_FLEET_TYP_CD = '320' then 1 else 0 end) as aircraft_320,
sum(case when PFL_FLEET_TYP_CD = '332' then 1 else 0 end) as aircraft_332,
sum(case when PFL_FLEET_TYP_CD = '333' then 1 else 0 end) as aircraft_333,
sum(case when PFL_FLEET_TYP_CD = '717' then 1 else 0 end) as aircraft_717,
sum(case when PFL_FLEET_TYP_CD = '738' then 1 else 0 end) as aircraft_738,
sum(case when PFL_FLEET_TYP_CD = '739' then 1 else 0 end) as aircraft_739,
sum(case when PFL_FLEET_TYP_CD = '73W' then 1 else 0 end) as aircraft_73W,
sum(case when PFL_FLEET_TYP_CD = '744' then 1 else 0 end) as aircraft_744,
sum(case when PFL_FLEET_TYP_CD = '753' then 1 else 0 end) as aircraft_753,
sum(case when PFL_FLEET_TYP_CD = '757' then 1 else 0 end) as aircraft_757,
sum(case when PFL_FLEET_TYP_CD = '763' then 1 else 0 end) as aircraft_763,
sum(case when PFL_FLEET_TYP_CD = '764' then 1 else 0 end) as aircraft_764,
sum(case when PFL_FLEET_TYP_CD = '76G' then 1 else 0 end) as aircraft_76G,
sum(case when PFL_FLEET_TYP_CD = '76L' then 1 else 0 end) as aircraft_76L,
sum(case when PFL_FLEET_TYP_CD = '772' then 1 else 0 end) as aircraft_772,
sum(case when PFL_FLEET_TYP_CD = '77L' then 1 else 0 end) as aircraft_77L,
sum(case when PFL_FLEET_TYP_CD = 'CPJ' then 1 else 0 end) as aircraft_CPJ,
sum(case when PFL_FLEET_TYP_CD = 'CR2' then 1 else 0 end) as aircraft_CR2,
sum(case when PFL_FLEET_TYP_CD = 'CR7' then 1 else 0 end) as aircraft_CR7,
sum(case when PFL_FLEET_TYP_CD = 'CR9' then 1 else 0 end) as aircraft_CR9,
sum(case when PFL_FLEET_TYP_CD = 'CRJ' then 1 else 0 end) as aircraft_CRJ,
sum(case when PFL_FLEET_TYP_CD = 'D95' then 1 else 0 end) as aircraft_D95,
sum(case when PFL_FLEET_TYP_CD = 'E70' then 1 else 0 end) as aircraft_E70,
sum(case when PFL_FLEET_TYP_CD = 'E75' then 1 else 0 end) as aircraft_E75,
sum(case when PFL_FLEET_TYP_CD = 'EC5' then 1 else 0 end) as aircraft_EC5,
sum(case when PFL_FLEET_TYP_CD = 'ERJ' then 1 else 0 end) as aircraft_ERJ,
sum(case when PFL_FLEET_TYP_CD = 'M90' then 1 else 0 end) as aircraft_M90,
sum(case when PFL_FLEET_TYP_CD = 'MD8' then 1 else 0 end) as aircraft_MD8,
sum(case when gts_1 = 1 then 1 else 0 end) as num_gts_1_1,
sum(case when gts_1 = 2 then 1 else 0 end) as num_gts_1_2,
sum(case when gts_1 = 3 then 1 else 0 end) as num_gts_1_3,
sum(case when gts_1 = 4 then 1 else 0 end) as num_gts_1_4,
sum(case when gts_1 = 5 then 1 else 0 end) as num_gts_1_5,
sum(case when gts_1 = . then 1 else 0 end) as num_gts_1_missing,
sum(case when gts_12 = 1 then 1 else 0 end) as num_gts_12_1,
sum(case when gts_12 = 2 then 1 else 0 end) as num_gts_12_2,
sum(case when gts_12 = 3 then 1 else 0 end) as num_gts_12_3,
sum(case when gts_12 = 4 then 1 else 0 end) as num_gts_12_4,
sum(case when gts_12 = 5 then 1 else 0 end) as num_gts_12_5,
sum(case when gts_12 = . then 1 else 0 end) as num_gts_12_missing,
sum(case when gts_13 = 1 then 1 else 0 end) as num_gts_13_1,
sum(case when gts_13 = 2 then 1 else 0 end) as num_gts_13_2,
sum(case when gts_13 = 3 then 1 else 0 end) as num_gts_13_3,
sum(case when gts_13 = 4 then 1 else 0 end) as num_gts_13_4,
sum(case when gts_13 = 5 then 1 else 0 end) as num_gts_13_5,
sum(case when gts_13 = . then 1 else 0 end) as num_gts_13_missing,
sum(case when gts_14 = 1 then 1 else 0 end) as num_gts_14_1,
sum(case when gts_15 = 1 then 1 else 0 end) as num_gts_15_1,
count(id) as num_recs
from p.final_clusters2
group by
cluster;
quit;
%mend desc;
%desc;

proc transpose data = descriptions out = p.six_cluster_descriptions(drop = _LABEL_ rename = (col1 = cluster1 col2 = cluster2
                                                                                              col3 = cluster3 col4 = cluster4
                                                                                              col5 = cluster5 col6 = cluster6));
format cluster1 cluster2 cluster3 cluster4 cluster5 cluster6 14.3;
run;

proc print data = p.six_cluster_descriptions noobs; run;
