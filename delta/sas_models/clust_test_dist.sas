%macro distancex;
distance1=sum(0
              ,(Lesiure_tkts_yr1-2.9501872525)**2
              ,(Business_tkts_yr1-2.617269772)**2
              ,(total_ap_yr1-1.9422255441)**2
              ,(int_tkts_yr1-0.2964531288)**2
              ,(dom_tkts_yr1-3.3560700582)**2
              ,(dl_miles_flown_yr1-2.6345854647)**2
              ,(dl_segs_yr1-3.2061096182)**2
              ,(dl_seg_rev_yr1-2.9513179099)**2
              ,(F_tkts_yr1-3.8255474952)**2
              ,(F_tkt_rev_yr1-3.6818382881)**2
              ,(F_miles_flown_yr1-3.6116366589)**2
              ,(C_tkts_yr1-1.1656432302)**2
              ,(C_tkt_rev_yr1-0.4865571749)**2
              ,(C_miles_flown_yr1-0.6458365952)**2
              ,(Y_tkts_yr1-1.7759079907)**2
              ,(Y_tkt_rev_yr1-1.606895912)**2
              ,(Y_miles_flown_yr1-1.1560865441)**2
              ,(F_upsell_tkts_yr1-2.0807541369)**2
              ,(F_upsell_tkt_rev_yr1-2.078611999)**2
              ,(F_upsell_miles_flown_yr1-1.9655619149)**2
              ,(total_miles_flown_yr1-2.5193999803)**2
              ,(total_tkts_yr1-3.3130340153)**2
              ,(total_tkt_rev_yr1-2.826253219)**2
              ,(dist_partners-1.1420340894)**2
              ,(tenure-0.4774154009)**2
              ,(DSC_Entries-3.0058708237)**2
              ,(hhi-0.2303832505)**2
              ,(lei_bus_ratio-0.7712375985)**2
              ,(dom_int_ratio-1.9836912382)**2
              ,(pct_trav_leis--0.79087369446215)**2
              ,(avg_rev_per_tkt-0.0611657025)**2
              ,(dl_rev_per_mile-0.668712564)**2
              ,(f_rev_per_mile-0.1548960302)**2
              ,(f_miles_per_tkt--0.33475604478595)**2
              ,(c_rev_per_mile--0.0805415346816)**2
              ,(c_miles_per_tkt--0.69648652410612)**2
              ,(y_rev_per_mile-0.6091531786)**2
              ,(y_miles_per_tkt--0.33124953464266)**2
              ,(f_upsell_rev_per_mile-0.1999181843)**2
              ,(f_upsell_miles_per_tkt--0.23069614032373)**2
              ,(pct_tkts_f-1.661233038)**2
              ,(pct_tkts_f_upsold--0.45330982062438)**2
              ,(pct_tkts_c-0.0861422624)**2
              ,(pct_tkts_y--1.52955703764472)**2
              ,(pct_miles_flown_f-1.5719243269)**2
              ,(pct_miles_f_upsold--0.42774793659673)**2
              ,(pct_miles_flown_c-0.2070050778)**2
              ,(pct_miles_flown_y--1.44065393022323)**2
              ,(pct_tkt_rev_f-1.5733283081)**2
              ,(pct_tkt_rev_f_upsold--0.42259456207823)**2
              ,(pct_tkt_rev_c-0.1770452617)**2
              ,(pct_tkt_rev_y--1.42281030455648)**2
              ,(pct_dl_ttl_miles-0.1350333396)**2
              ,(pct_dl_ttl_rev-0.1391020601)**2
              ,(dsc_entry_per_trip--0.15367025110535)**2
              ,(rto_Fmiles_Cmiles-1.665164138)**2
              ,(rto_Fmiles_Ymiles-1.1211286741)**2
              ,(rto_Fmiles_F_upsell_miles-1.2404717715)**2
              ,(rto_Cmiles_Ymiles-0.1500838074)**2
              ,(rto_Cmiles_F_upsell_miles-0.0512655951)**2
              ,(rto_Ymiles_F_upsell_miles-0.3430046738)**2
              )**(1/2);
distance2=sum(0
              ,(Lesiure_tkts_yr1--0.42693766660373)**2
              ,(Business_tkts_yr1--0.33070137172512)**2
              ,(total_ap_yr1--0.35601082892037)**2
              ,(int_tkts_yr1--0.22845450581758)**2
              ,(dom_tkts_yr1--0.43123268855605)**2
              ,(dl_miles_flown_yr1--0.46487747810638)**2
              ,(dl_segs_yr1--0.44956830041035)**2
              ,(dl_seg_rev_yr1--0.47152177371101)**2
              ,(F_tkts_yr1--0.38398642864679)**2
              ,(F_tkt_rev_yr1--0.3776501158947)**2
              ,(F_miles_flown_yr1--0.39535782529357)**2
              ,(C_tkts_yr1--0.26426307704748)**2
              ,(C_tkt_rev_yr1--0.21941554004514)**2
              ,(C_miles_flown_yr1--0.2415853768367)**2
              ,(Y_tkts_yr1--0.36368357057045)**2
              ,(Y_tkt_rev_yr1--0.37082314482885)**2
              ,(Y_miles_flown_yr1--0.33211998628768)**2
              ,(F_upsell_tkts_yr1--0.25178066648438)**2
              ,(F_upsell_tkt_rev_yr1--0.25778794529576)**2
              ,(F_upsell_miles_flown_yr1--0.26927821192654)**2
              ,(total_miles_flown_yr1--0.46325869634429)**2
              ,(total_tkts_yr1--0.45818640210069)**2
              ,(total_tkt_rev_yr1--0.4716605561912)**2
              ,(dist_partners--0.29444330993515)**2
              ,(tenure--0.15895911861076)**2
              ,(DSC_Entries--0.2868162264615)**2
              ,(hhi--0.08677578843952)**2
              ,(lei_bus_ratio--0.0946782387745)**2
              ,(dom_int_ratio--0.17024077275773)**2
              ,(pct_trav_leis-0.1915088752)**2
              ,(avg_rev_per_tkt--0.21794471557039)**2
              ,(dl_rev_per_mile--0.23760165774856)**2
              ,(f_rev_per_mile--0.00776100848862)**2
              ,(f_miles_per_tkt--0.01464681744156)**2
              ,(c_rev_per_mile--0.01237747485387)**2
              ,(c_miles_per_tkt--0.01629028828183)**2
              ,(y_rev_per_mile--0.1286959757934)**2
              ,(y_miles_per_tkt-0.0270058064)**2
              ,(f_upsell_rev_per_mile-0.0030560757)**2
              ,(f_upsell_miles_per_tkt--0.01076672814782)**2
              ,(pct_tkts_f--0.52513320296289)**2
              ,(pct_tkts_f_upsold--0.03306259786683)**2
              ,(pct_tkts_c--0.22802149299667)**2
              ,(pct_tkts_y-0.5623016548)**2
              ,(pct_miles_flown_f--0.48881189710193)**2
              ,(pct_miles_f_upsold--0.03364274273557)**2
              ,(pct_miles_flown_c--0.26134334214329)**2
              ,(pct_miles_flown_y-0.5477019526)**2
              ,(pct_tkt_rev_f--0.5051744419425)**2
              ,(pct_tkt_rev_f_upsold--0.03346478867186)**2
              ,(pct_tkt_rev_c--0.26422428401191)**2
              ,(pct_tkt_rev_y-0.5680877995)**2
              ,(pct_dl_ttl_miles-0.047061288)**2
              ,(pct_dl_ttl_rev-0.0532521061)**2
              ,(dsc_entry_per_trip--0.02662687633462)**2
              ,(rto_Fmiles_Cmiles--0.01064887799224)**2
              ,(rto_Fmiles_Ymiles--0.12876099453768)**2
              ,(rto_Fmiles_F_upsell_miles--0.00322587467254)**2
              ,(rto_Cmiles_Ymiles--0.08362740587315)**2
              ,(rto_Cmiles_F_upsell_miles--0.00256996914785)**2
              ,(rto_Ymiles_F_upsell_miles-0.0119695857)**2
              )**(1/2);
distance3=sum(0
              ,(Lesiure_tkts_yr1-0.4132473854)**2
              ,(Business_tkts_yr1-0.1136092271)**2
              ,(total_ap_yr1-0.5116317218)**2
              ,(int_tkts_yr1-1.835121913)**2
              ,(dom_tkts_yr1-0.0401166706)**2
              ,(dl_miles_flown_yr1-1.073012703)**2
              ,(dl_segs_yr1-0.3340185971)**2
              ,(dl_seg_rev_yr1-1.4434138433)**2
              ,(F_tkts_yr1-0.2437539453)**2
              ,(F_tkt_rev_yr1-0.2961167987)**2
              ,(F_miles_flown_yr1-0.3278771452)**2
              ,(C_tkts_yr1-2.5751733807)**2
              ,(C_tkt_rev_yr1-3.1301563024)**2
              ,(C_miles_flown_yr1-3.122299338)**2
              ,(Y_tkts_yr1--0.10959359567456)**2
              ,(Y_tkt_rev_yr1-0.0999380115)**2
              ,(Y_miles_flown_yr1-0.1197593818)**2
              ,(F_upsell_tkts_yr1-0.1366836233)**2
              ,(F_upsell_tkt_rev_yr1-0.186237003)**2
              ,(F_upsell_miles_flown_yr1-0.2134506084)**2
              ,(total_miles_flown_yr1-1.1875028048)**2
              ,(total_tkts_yr1-0.3521441481)**2
              ,(total_tkt_rev_yr1-1.6405323407)**2
              ,(dist_partners-0.480198583)**2
              ,(tenure-0.3900473672)**2
              ,(DSC_Entries-0.5836097507)**2
              ,(hhi-0.288541962)**2
              ,(lei_bus_ratio-0.1838352069)**2
              ,(dom_int_ratio--0.45287935731686)**2
              ,(pct_trav_leis--0.03022088137457)**2
              ,(avg_rev_per_tkt-2.4101582691)**2
              ,(dl_rev_per_mile-0.842062362)**2
              ,(f_rev_per_mile--0.00123151373267)**2
              ,(f_miles_per_tkt-0.1234919178)**2
              ,(c_rev_per_mile-1.1121127002)**2
              ,(c_miles_per_tkt-2.0617550324)**2
              ,(y_rev_per_mile-0.1652632778)**2
              ,(y_miles_per_tkt-0.1704062315)**2
              ,(f_upsell_rev_per_mile-0.0089184873)**2
              ,(f_upsell_miles_per_tkt-0.1873937393)**2
              ,(pct_tkts_f-0.3309598572)**2
              ,(pct_tkts_f_upsold--0.1302950608568)**2
              ,(pct_tkts_c-3.2272917136)**2
              ,(pct_tkts_y--1.56441717717608)**2
              ,(pct_miles_flown_f--0.01462098270972)**2
              ,(pct_miles_f_upsold--0.10897890219063)**2
              ,(pct_miles_flown_c-3.692532265)**2
              ,(pct_miles_flown_y--1.85679240907538)**2
              ,(pct_tkt_rev_f--0.11775815143679)**2
              ,(pct_tkt_rev_f_upsold--0.09822366608079)**2
              ,(pct_tkt_rev_c-3.8221578428)**2
              ,(pct_tkt_rev_y--1.95096094828554)**2
              ,(pct_dl_ttl_miles--0.77382511262451)**2
              ,(pct_dl_ttl_rev--0.97007472011856)**2
              ,(dsc_entry_per_trip--0.22234264636576)**2
              ,(rto_Fmiles_Cmiles--1.02002580335285)**2
              ,(rto_Fmiles_Ymiles-0.1921846585)**2
              ,(rto_Fmiles_F_upsell_miles-0.007159539)**2
              ,(rto_Cmiles_Ymiles-1.2083056864)**2
              ,(rto_Cmiles_F_upsell_miles-0.9969521149)**2
              ,(rto_Ymiles_F_upsell_miles--0.07428083002957)**2
              )**(1/2);
distance4=sum(0
              ,(Lesiure_tkts_yr1--0.07629071784021)**2
              ,(Business_tkts_yr1--0.15265694633071)**2
              ,(total_ap_yr1--0.05885906678056)**2
              ,(int_tkts_yr1--0.23107754929196)**2
              ,(dom_tkts_yr1--0.08624137923119)**2
              ,(dl_miles_flown_yr1--0.18639931606185)**2
              ,(dl_segs_yr1--0.14124773747717)**2
              ,(dl_seg_rev_yr1--0.0522397074419)**2
              ,(F_tkts_yr1-0.2545109945)**2
              ,(F_tkt_rev_yr1-0.4189264715)**2
              ,(F_miles_flown_yr1-0.3619663132)**2
              ,(C_tkts_yr1--0.14758245836626)**2
              ,(C_tkt_rev_yr1--0.15690093631599)**2
              ,(C_miles_flown_yr1--0.16988548805216)**2
              ,(Y_tkts_yr1--0.42184759802909)**2
              ,(Y_tkt_rev_yr1--0.40743560590796)**2
              ,(Y_miles_flown_yr1--0.47059171615242)**2
              ,(F_upsell_tkts_yr1-0.9070417874)**2
              ,(F_upsell_tkt_rev_yr1-0.9346910665)**2
              ,(F_upsell_miles_flown_yr1-1.0397743863)**2
              ,(total_miles_flown_yr1--0.20397052467863)**2
              ,(total_tkts_yr1--0.12326678430191)**2
              ,(total_tkt_rev_yr1--0.07293632251023)**2
              ,(dist_partners-0.0422940977)**2
              ,(tenure-0.1785293113)**2
              ,(DSC_Entries--0.08927941210061)**2
              ,(hhi-0.1144158749)**2
              ,(lei_bus_ratio--0.02450751652379)**2
              ,(dom_int_ratio-0.0029216243)**2
              ,(pct_trav_leis--0.00411130406404)**2
              ,(avg_rev_per_tkt-0.176263806)**2
              ,(dl_rev_per_mile-0.62084206)**2
              ,(f_rev_per_mile-0.2607997371)**2
              ,(f_miles_per_tkt-0.342269103)**2
              ,(c_rev_per_mile-0.0231747918)**2
              ,(c_miles_per_tkt--0.11858809573011)**2
              ,(y_rev_per_mile-0.2138139419)**2
              ,(y_miles_per_tkt--0.16829653258607)**2
              ,(f_upsell_rev_per_mile--0.17694844032868)**2
              ,(f_upsell_miles_per_tkt-0.1502441962)**2
              ,(pct_tkts_f-1.8703541344)**2
              ,(pct_tkts_f_upsold-1.9968324073)**2
              ,(pct_tkts_c--0.10873660814126)**2
              ,(pct_tkts_y--1.64137448379344)**2
              ,(pct_miles_flown_f-1.9875348214)**2
              ,(pct_miles_f_upsold-1.9851366495)**2
              ,(pct_miles_flown_c--0.12156240622899)**2
              ,(pct_miles_flown_y--1.62752519505294)**2
              ,(pct_tkt_rev_f-2.0555158521)**2
              ,(pct_tkt_rev_f_upsold-1.9654470936)**2
              ,(pct_tkt_rev_c--0.13050702068183)**2
              ,(pct_tkt_rev_y--1.66477327376587)**2
              ,(pct_dl_ttl_miles-0.177734643)**2
              ,(pct_dl_ttl_rev-0.1986870086)**2
              ,(dsc_entry_per_trip--0.04500748079393)**2
              ,(rto_Fmiles_Cmiles--0.00371973761315)**2
              ,(rto_Fmiles_Ymiles-0.3162747963)**2
              ,(rto_Fmiles_F_upsell_miles--0.61933949522452)**2
              ,(rto_Cmiles_Ymiles--0.01725263892636)**2
              ,(rto_Cmiles_F_upsell_miles--0.43334913960845)**2
              ,(rto_Ymiles_F_upsell_miles--0.62961227468977)**2
              )**(1/2);
distance5=sum(0
              ,(Lesiure_tkts_yr1-0.8039589405)**2
              ,(Business_tkts_yr1-0.6645274458)**2
              ,(total_ap_yr1-0.7369484316)**2
              ,(int_tkts_yr1-0.369727858)**2
              ,(dom_tkts_yr1-0.8416798618)**2
              ,(dl_miles_flown_yr1-0.8892841478)**2
              ,(dl_segs_yr1-0.8906910951)**2
              ,(dl_seg_rev_yr1-0.6510784478)**2
              ,(F_tkts_yr1-0.2947122669)**2
              ,(F_tkt_rev_yr1-0.1883224925)**2
              ,(F_miles_flown_yr1-0.2943377252)**2
              ,(C_tkts_yr1-0.0349920401)**2
              ,(C_tkt_rev_yr1--0.12874773945746)**2
              ,(C_miles_flown_yr1--0.07431952711314)**2
              ,(Y_tkts_yr1-1.2062673351)**2
              ,(Y_tkt_rev_yr1-1.2009139025)**2
              ,(Y_miles_flown_yr1-1.196183183)**2
              ,(F_upsell_tkts_yr1--0.15681604955662)**2
              ,(F_upsell_tkt_rev_yr1--0.16590632014995)**2
              ,(F_upsell_miles_flown_yr1--0.1708319480381)**2
              ,(total_miles_flown_yr1-0.8871438375)**2
              ,(total_tkts_yr1-0.8812904763)**2
              ,(total_tkt_rev_yr1-0.6354493071)**2
              ,(dist_partners-0.6429282268)**2
              ,(tenure-0.2451918848)**2
              ,(DSC_Entries-0.2409543524)**2
              ,(hhi-0.1083366182)**2
              ,(lei_bus_ratio-0.1294457509)**2
              ,(dom_int_ratio-0.2957527982)**2
              ,(pct_trav_leis--0.5036426478354)**2
              ,(avg_rev_per_tkt--0.03210982946635)**2
              ,(dl_rev_per_mile-0.084090457)**2
              ,(f_rev_per_mile--0.16685757700581)**2
              ,(f_miles_per_tkt--0.11651078673355)**2
              ,(c_rev_per_mile--0.27449048872301)**2
              ,(c_miles_per_tkt--0.30913282837734)**2
              ,(y_rev_per_mile-0.1490275395)**2
              ,(y_miles_per_tkt-0.0311989985)**2
              ,(f_upsell_rev_per_mile-0.0493377702)**2
              ,(f_upsell_miles_per_tkt--0.05505241153898)**2
              ,(pct_tkts_f-0.2834532008)**2
              ,(pct_tkts_f_upsold--0.96433643654129)**2
              ,(pct_tkts_c--0.06403305064435)**2
              ,(pct_tkts_y--0.2300905092026)**2
              ,(pct_miles_flown_f-0.2005716939)**2
              ,(pct_miles_f_upsold--0.96708146207832)**2
              ,(pct_miles_flown_c--0.09810209660312)**2
              ,(pct_miles_flown_y--0.12079055816642)**2
              ,(pct_tkt_rev_f-0.2479376825)**2
              ,(pct_tkt_rev_f_upsold--0.95994646386732)**2
              ,(pct_tkt_rev_c--0.11303603381782)**2
              ,(pct_tkt_rev_y--0.14860969095097)**2
              ,(pct_dl_ttl_miles--0.08559411864171)**2
              ,(pct_dl_ttl_rev--0.06475974971991)**2
              ,(dsc_entry_per_trip-0.2240215183)**2
              ,(rto_Fmiles_Cmiles--0.0419394326955)**2
              ,(rto_Fmiles_Ymiles--0.03818891998355)**2
              ,(rto_Fmiles_F_upsell_miles-0.1069450728)**2
              ,(rto_Cmiles_Ymiles--0.07157148188029)**2
              ,(rto_Cmiles_F_upsell_miles--0.02643366902301)**2
              ,(rto_Ymiles_F_upsell_miles-0.2864002371)**2
              )**(1/2);
%mend distancex;
