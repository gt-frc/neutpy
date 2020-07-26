import numpy as np
from scipy.interpolate import Rbf


def _calc_svc31_degas():
    """ Calculates cross section for "Continuum n=3 / n=1," whatever that is.

    accepts ne in m^-3 and Te in eV
    results in erg/s, i.e. units of power.
    """

    # populate the 15x60 array to be used for interpolations
    # rows correspond to ne, columns correspond to Te

    c31_data = np.zeros((15, 60))
    c31_data[0] = np.array([1.22880E-09, 5.96713E-10, 3.15073E-10, 1.80040E-10, 1.10354E-10, 7.18020E-11,
                            4.90724E-11, 3.48876E-11, 2.55822E-11, 1.92070E-11, 1.46752E-11, 1.13548E-11,
                            8.86129E-12, 6.95280E-12, 5.47148E-12, 4.30986E-12, 3.39310E-12, 2.66732E-12,
                            2.09151E-12, 1.63516E-12, 1.27415E-12, 9.89251E-13, 7.65227E-13, 5.89712E-13,
                            4.52767E-13, 3.46360E-13, 2.64028E-13, 2.00586E-13, 1.51896E-13, 1.14673E-13,
                            8.63202E-14, 6.48003E-14, 4.85202E-14, 3.62426E-14, 2.70104E-14, 2.00872E-14,
                            1.49088E-14, 1.10448E-14, 8.16795E-15, 6.03057E-15, 4.44567E-15, 3.27258E-15,
                            2.40578E-15, 1.76632E-15, 1.29528E-15, 9.48789E-16, 6.94246E-16, 5.07482E-16,
                            3.70609E-16, 2.70409E-16, 1.97132E-16, 1.43596E-16, 1.04518E-16, 7.60196E-17,
                            5.52532E-17, 4.01330E-17, 2.91321E-17, 2.11338E-17, 1.53227E-17, 1.11033E-17])
    c31_data[1] = np.array([9.21089E-09, 3.99038E-09, 1.88499E-09, 9.71730E-10, 5.43476E-10, 3.26718E-10,
                            2.08830E-10, 1.40410E-10, 9.83396E-11, 7.11159E-11, 5.27066E-11, 3.97877E-11,
                            3.04374E-11, 2.35003E-11, 1.82542E-11, 1.42281E-11, 1.11063E-11, 8.67009E-12,
                            6.75982E-12, 5.26009E-12, 4.08305E-12, 3.15990E-12, 2.43789E-12, 1.87459E-12,
                            1.43666E-12, 1.09736E-12, 8.35456E-13, 6.34043E-13, 4.79716E-13, 3.61893E-13,
                            2.72249E-13, 2.04273E-13, 1.52887E-13, 1.14160E-13, 8.50548E-14, 6.32386E-14,
                            4.69268E-14, 3.47589E-14, 2.57021E-14, 1.89745E-14, 1.39868E-14, 1.02956E-14,
                            7.56843E-15, 5.55668E-15, 4.07486E-15, 2.98489E-15, 2.18418E-15, 1.59668E-15,
                            1.16611E-15, 8.50897E-16, 6.20369E-16, 4.51935E-16, 3.28983E-16, 2.39308E-16,
                            1.73958E-16, 1.26371E-16, 9.17438E-17, 6.65653E-17, 4.82693E-17, 3.49829E-17])
    c31_data[2] = np.array([7.94866E-08, 3.09013E-08, 1.30188E-08, 5.99741E-09, 3.01984E-09, 1.65305E-09,
                            9.73693E-10, 6.10651E-10, 4.03494E-10, 2.78034E-10, 1.98027E-10, 1.44695E-10,
                            1.07781E-10, 8.14250E-11, 6.21364E-11, 4.77370E-11, 3.68266E-11, 2.84727E-11,
                            2.20253E-11, 1.70281E-11, 1.31474E-11, 1.01297E-11, 7.78658E-12, 5.96895E-12,
                            4.56292E-12, 3.47791E-12, 2.64320E-12, 2.00304E-12, 1.51366E-12, 1.14074E-12,
                            8.57451E-13, 6.42917E-13, 4.80917E-13, 3.58931E-13, 2.67320E-13, 1.98691E-13,
                            1.47404E-13, 1.09162E-13, 8.07064E-14, 5.95748E-14, 4.39113E-14, 3.23211E-14,
                            2.37589E-14, 1.74435E-14, 1.27919E-14, 9.37046E-15, 6.85705E-15, 5.01289E-15,
                            3.66132E-15, 2.67182E-15, 1.94811E-15, 1.41932E-15, 1.03328E-15, 7.51708E-16,
                            5.46493E-16, 3.97044E-16, 2.88286E-16, 2.09196E-16, 1.51718E-16, 1.09973E-16])
    c31_data[3] = np.array([8.10718E-07, 2.86603E-07, 1.08050E-07, 4.42004E-08, 1.97587E-08, 9.69321E-09,
                            5.16911E-09, 2.97107E-09, 1.82295E-09, 1.17985E-09, 7.97366E-10, 5.57669E-10,
                            4.00553E-10, 2.93605E-10, 2.18533E-10, 1.64473E-10, 1.24752E-10, 9.51147E-11,
                            7.27370E-11, 5.57044E-11, 4.26737E-11, 3.26664E-11, 2.49754E-11, 1.90592E-11,
                            1.45153E-11, 1.10291E-11, 8.36035E-12, 6.32197E-12, 4.76890E-12, 3.58871E-12,
                            2.69424E-12, 2.01819E-12, 1.50846E-12, 1.12512E-12, 8.37527E-13, 6.22255E-13,
                            4.61494E-13, 3.41685E-13, 2.52575E-13, 1.86421E-13, 1.37398E-13, 1.01130E-13,
                            7.43402E-14, 5.45814E-14, 4.00287E-14, 2.93245E-14, 2.14608E-14, 1.56908E-14,
                            1.14616E-14, 8.36503E-15, 6.10005E-15, 4.44487E-15, 3.23639E-15, 2.35481E-15,
                            1.71221E-15, 1.24416E-15, 9.03507E-16, 6.55738E-16, 4.75647E-16, 3.44830E-16])
    c31_data[4] = np.array([1.00830E-05, 3.32356E-06, 1.13948E-06, 4.15227E-07, 1.63261E-07, 7.07967E-08,
                            3.35173E-08, 1.72630E-08, 9.63446E-09, 5.74327E-09, 3.61581E-09, 2.37973E-09,
                            1.62288E-09, 1.13822E-09, 8.16237E-10, 5.95419E-10, 4.39967E-10, 3.28199E-10,
                            2.46464E-10, 1.85920E-10, 1.40648E-10, 1.06547E-10, 8.07559E-11, 6.11819E-11,
                            4.63154E-11, 3.50144E-11, 2.64312E-11, 1.99181E-11, 1.49822E-11, 1.12482E-11,
                            8.42861E-12, 6.30462E-12, 4.70673E-12, 3.50742E-12, 2.60908E-12, 1.93737E-12,
                            1.43630E-12, 1.06314E-12, 7.85756E-13, 5.79915E-13, 4.27416E-13, 3.14612E-13,
                            2.31296E-13, 1.69845E-13, 1.24583E-13, 9.12869E-14, 6.68226E-14, 4.88682E-14,
                            3.57057E-14, 2.60661E-14, 1.90133E-14, 1.38581E-14, 1.00931E-14, 7.34582E-15,
                            5.34271E-15, 3.88331E-15, 2.82082E-15, 2.04782E-15, 1.48580E-15, 1.07745E-15])
    c31_data[5] = np.array([1.37049E-04, 4.41292E-05, 1.44100E-05, 4.85864E-06, 1.72186E-06, 6.65710E-07,
                            2.78612E-07, 1.26775E-07, 6.33420E-08, 3.41834E-08, 1.96774E-08, 1.19545E-08,
                            7.59326E-09, 5.00049E-09, 3.39412E-09, 2.36073E-09, 1.67427E-09, 1.20577E-09,
                            8.78720E-10, 6.46179E-10, 4.78388E-10, 3.55860E-10, 2.65609E-10, 1.98653E-10,
                            1.48762E-10, 1.11448E-10, 8.34923E-11, 6.25246E-11, 4.67866E-11, 3.49770E-11,
                            2.61186E-11, 1.94904E-11, 1.45221E-11, 1.08060E-11, 8.03020E-12, 5.95790E-12,
                            4.41501E-12, 3.26727E-12, 2.41477E-12, 1.78243E-12, 1.31405E-12, 9.67595E-13,
                            7.11664E-13, 5.22848E-13, 3.83720E-13, 2.81326E-13, 2.06053E-13, 1.50780E-13,
                            1.10235E-13, 8.05233E-14, 5.87717E-14, 4.28621E-14, 3.12358E-14, 2.27468E-14,
                            1.65535E-14, 1.20385E-14, 8.74945E-15, 6.35516E-15, 4.61339E-15, 3.34712E-15])
    c31_data[6] = np.array([1.57644E-03, 5.17160E-04, 1.69140E-04, 5.56930E-05, 1.87404E-05, 6.72829E-06,
                            2.57005E-06, 1.05659E-06, 4.78953E-07, 2.35601E-07, 1.24156E-07, 6.94133E-08,
                            4.08051E-08, 2.49992E-08, 1.58915E-08, 1.04188E-08, 7.00791E-09, 4.81392E-09,
                            3.36402E-09, 2.38370E-09, 1.70804E-09, 1.23475E-09, 8.98892E-10, 6.57896E-10,
                            4.83522E-10, 3.56440E-10, 2.63355E-10, 1.94905E-10, 1.44391E-10, 1.07039E-10,
                            7.93688E-11, 5.89520E-11, 4.37486E-11, 3.24574E-11, 2.40704E-11, 1.78278E-11,
                            1.32001E-11, 9.76545E-12, 7.21845E-12, 5.33090E-12, 3.93323E-12, 2.89923E-12,
                            2.13498E-12, 1.57067E-12, 1.15441E-12, 8.47651E-13, 6.21826E-13, 4.55744E-13,
                            3.33722E-13, 2.44156E-13, 1.78478E-13, 1.30359E-13, 9.51385E-14, 6.93804E-14,
                            5.05587E-14, 3.68168E-14, 2.67915E-14, 1.94833E-14, 1.41596E-14, 1.02844E-14])
    c31_data[7] = np.array([1.29414E-02, 4.41726E-03, 1.48932E-03, 4.97549E-04, 1.66369E-04, 5.80229E-05,
                            2.10666E-05, 8.09309E-06, 3.40132E-06, 1.54674E-06, 7.54293E-07, 3.91896E-07,
                            2.15254E-07, 1.23775E-07, 7.43317E-08, 4.63009E-08, 2.97309E-08, 1.95766E-08,
                            1.31596E-08, 8.99748E-09, 6.23814E-09, 4.37464E-09, 3.09675E-09, 2.20897E-09,
                            1.58565E-09, 1.14404E-09, 8.28902E-10, 6.02712E-10, 4.39455E-10, 3.21166E-10,
                            2.35147E-10, 1.72984E-10, 1.27265E-10, 9.37542E-11, 6.91438E-11, 5.09487E-11,
                            3.75963E-11, 2.77501E-11, 2.04860E-11, 1.51230E-11, 1.11621E-11, 8.23633E-12,
                            6.07504E-12, 4.47874E-12, 3.30000E-12, 2.42994E-12, 1.78804E-12, 1.31473E-12,
                            9.65964E-13, 7.09150E-13, 5.20187E-13, 3.81260E-13, 2.79204E-13, 2.04297E-13,
                            1.49363E-13, 1.09112E-13, 7.96447E-14, 5.80899E-14, 4.23364E-14, 3.08323E-14])
    c31_data[8] = np.array([7.09915E-02, 2.54411E-02, 8.96603E-03, 3.10441E-03, 1.06066E-03, 3.69645E-04,
                            1.30522E-04, 4.76788E-05, 1.87975E-05, 7.99304E-06, 3.66032E-06, 1.79938E-06,
                            9.42370E-07, 5.19773E-07, 3.01367E-07, 1.82112E-07, 1.13839E-07, 7.31531E-08,
                            4.80757E-08, 3.21774E-08, 2.18593E-08, 1.50311E-08, 1.04392E-08, 7.30962E-09,
                            5.15300E-09, 3.65305E-09, 2.60187E-09, 1.86073E-09, 1.33511E-09, 9.60752E-10,
                            6.93067E-10, 5.02826E-10, 3.65121E-10, 2.65741E-10, 1.93836E-10, 1.41274E-10,
                            1.03284E-10, 7.56138E-11, 5.54283E-11, 4.06770E-11, 2.98812E-11, 2.19696E-11,
                            1.61644E-11, 1.19005E-11, 8.76522E-12, 6.45811E-12, 4.75921E-12, 3.50750E-12,
                            2.58488E-12, 1.90464E-12, 1.40304E-12, 1.03315E-12, 7.60431E-13, 5.59396E-13,
                            4.11256E-13, 3.02142E-13, 2.21816E-13, 1.62720E-13, 1.19272E-13, 8.73536E-14])
    c31_data[9] = np.array([2.72666E-01, 1.02331E-01, 3.76277E-02, 1.35313E-02, 4.75784E-03, 1.67636E-03,
                            5.84444E-04, 2.06968E-04, 7.83906E-05, 3.20428E-05, 1.41836E-05, 6.78100E-06,
                            3.47119E-06, 1.87934E-06, 1.07297E-06, 6.39742E-07, 3.95085E-07, 2.51038E-07,
                            1.63226E-07, 1.08126E-07, 7.27122E-08, 4.94979E-08, 3.40317E-08, 2.35884E-08,
                            1.64588E-08, 1.15470E-08, 8.13787E-09, 5.75752E-09, 4.08637E-09, 2.90832E-09,
                            2.07477E-09, 1.48707E-09, 1.06709E-09, 7.67433E-10, 5.53137E-10, 3.98485E-10,
                            2.87949E-10, 2.08433E-10, 1.51102E-10, 1.09701E-10, 7.97586E-11, 5.80690E-11,
                            4.23340E-11, 3.09038E-11, 2.25867E-11, 1.65272E-11, 1.21066E-11, 8.87751E-12,
                            6.51590E-12, 4.78669E-12, 3.51913E-12, 2.58900E-12, 1.90582E-12, 1.40358E-12,
                            1.03407E-12, 7.62031E-13, 5.61632E-13, 4.13941E-13, 3.05057E-13, 2.24766E-13])
    c31_data[10] = np.array([8.22545E-01, 3.24432E-01, 1.24309E-01, 4.62657E-02, 1.67130E-02, 5.98163E-03,
                             2.08950E-03, 7.34765E-04, 2.75557E-04, 1.11671E-04, 4.90685E-05, 2.32796E-05,
                             1.18192E-05, 6.35489E-06, 3.60223E-06, 2.13253E-06, 1.30806E-06, 8.25860E-07,
                             5.33773E-07, 3.51582E-07, 2.35129E-07, 1.59193E-07, 1.08856E-07, 7.50350E-08,
                             5.20587E-08, 3.63090E-08, 2.54335E-08, 1.78795E-08, 1.26060E-08, 8.90996E-09,
                             6.31079E-09, 4.48413E-09, 3.19022E-09, 2.27367E-09, 1.62323E-09, 1.15927E-09,
                             8.29639E-10, 5.94691E-10, 4.26822E-10, 3.06746E-10, 2.20748E-10, 1.59073E-10,
                             1.14782E-10, 8.29417E-11, 6.00091E-11, 4.34739E-11, 3.15356E-11, 2.29048E-11,
                             1.66569E-11, 1.21281E-11, 8.84116E-12, 6.45254E-12, 4.71457E-12, 3.44847E-12,
                             2.52502E-12, 1.85070E-12, 1.35775E-12, 9.96995E-13, 7.32699E-13, 5.38876E-13])
    c31_data[11] = np.array([2.16525E+00, 8.92316E-01, 3.58603E-01, 1.38226E-01, 5.14668E-02, 1.88247E-02,
                             6.67227E-03, 2.37532E-03, 9.03093E-04, 3.70629E-04, 1.63790E-04, 7.75110E-05,
                             3.90869E-05, 2.08845E-05, 1.17630E-05, 6.92485E-06, 4.22780E-06, 2.65891E-06,
                             1.71283E-06, 1.12489E-06, 7.50256E-07, 5.06633E-07, 3.45542E-07, 2.37558E-07,
                             1.64366E-07, 1.14310E-07, 7.98272E-08, 5.59337E-08, 3.92981E-08, 2.76717E-08,
                             1.95206E-08, 1.37991E-08, 9.76607E-09, 6.92049E-09, 4.90977E-09, 3.48554E-09,
                             2.47736E-09, 1.76274E-09, 1.25536E-09, 8.94849E-10, 6.38486E-10, 4.56018E-10,
                             3.26029E-10, 2.33368E-10, 1.67214E-10, 1.19949E-10, 8.61452E-11, 6.19427E-11,
                             4.45949E-11, 3.21460E-11, 2.32018E-11, 1.67678E-11, 1.21336E-11, 8.79148E-12,
                             6.37804E-12, 4.63296E-12, 3.36950E-12, 2.45356E-12, 1.78869E-12, 1.30548E-12])
    c31_data[12] = np.array([4.88870E+00, 2.16049E+00, 9.34938E-01, 3.76958E-01, 1.45636E-01, 5.50373E-02,
                             2.00214E-02, 7.31364E-03, 2.86023E-03, 1.19931E-03, 5.33374E-04, 2.51350E-04,
                             1.25949E-04, 6.69915E-05, 3.76041E-05, 2.20846E-05, 1.34605E-05, 8.45474E-06,
                             5.44080E-06, 3.56995E-06, 2.37893E-06, 1.60506E-06, 1.09374E-06, 7.51232E-07,
                             5.19255E-07, 3.60728E-07, 2.51611E-07, 1.76069E-07, 1.23526E-07, 8.68442E-08,
                             6.11577E-08, 4.31344E-08, 3.04557E-08, 2.15247E-08, 1.52253E-08, 1.07760E-08,
                             7.63222E-09, 5.40905E-09, 3.83545E-09, 2.72106E-09, 1.93147E-09, 1.37173E-09,
                             9.74729E-10, 6.93104E-10, 4.93114E-10, 3.51058E-10, 2.50099E-10, 1.78307E-10,
                             1.27224E-10, 9.08530E-11, 6.49387E-11, 4.64609E-11, 3.32749E-11, 2.38569E-11,
                             1.71240E-11, 1.23058E-11, 8.85419E-12, 6.37881E-12, 4.60146E-12, 3.32374E-12])
    c31_data[13] = np.array([9.15993E+00, 4.46532E+00, 2.19690E+00, 9.38575E-01, 3.81294E-01, 1.50115E-01,
                             5.68247E-02, 2.14795E-02, 8.68494E-03, 3.72549E-03, 1.67124E-03, 7.88877E-04,
                             3.95622E-04, 2.10715E-04, 1.18442E-04, 6.96444E-05, 4.24890E-05, 2.67070E-05,
                             1.71948E-05, 1.12856E-05, 7.52150E-06, 5.07482E-06, 3.45786E-06, 2.37464E-06,
                             1.64099E-06, 1.13966E-06, 7.94643E-07, 5.55833E-07, 3.89776E-07, 2.73882E-07,
                             1.92758E-07, 1.35843E-07, 9.58322E-08, 6.76640E-08, 4.78087E-08, 3.37983E-08,
                             2.39054E-08, 1.69153E-08, 1.19734E-08, 8.47788E-09, 6.00459E-09, 4.25401E-09,
                             3.01458E-09, 2.13706E-09, 1.51530E-09, 1.07474E-09, 7.62502E-10, 5.41154E-10,
                             3.84197E-10, 2.72868E-10, 1.93880E-10, 1.37819E-10, 9.80179E-11, 6.97489E-11,
                             4.96625E-11, 3.53836E-11, 2.52280E-11, 1.80010E-11, 1.28550E-11, 9.18830E-12])
    c31_data[14] = np.array([1.57231E+01, 8.31557E+00, 4.17893E+00, 1.97722E+00, 9.15943E-01, 3.80513E-01,
                             1.50522E-01, 5.94129E-02, 2.49763E-02, 1.10109E-02, 5.02519E-03, 2.40069E-03,
                             1.21522E-03, 6.52153E-04, 3.68709E-04, 2.17755E-04, 1.33281E-04, 8.39733E-05,
                             5.41562E-05, 3.55873E-05, 2.37376E-05, 1.60251E-05, 1.09232E-05, 7.50309E-06,
                             5.18563E-06, 3.60158E-06, 2.51121E-06, 1.75642E-06, 1.23156E-06, 8.65250E-07,
                             6.08858E-07, 4.28974E-07, 3.02545E-07, 2.13550E-07, 1.50831E-07, 1.06588E-07,
                             7.53549E-08, 5.32920E-08, 3.76995E-08, 2.66754E-08, 1.88788E-08, 1.33632E-08,
                             9.46047E-09, 6.69916E-09, 4.74417E-09, 3.36013E-09, 2.38018E-09, 1.68626E-09,
                             1.19481E-09, 8.46714E-10, 6.00130E-10, 4.25432E-10, 3.01646E-10, 2.13923E-10,
                             1.51746E-10, 1.07668E-10, 7.64156E-11, 5.42517E-11, 3.85297E-11, 2.73744E-11])

    Te_vals, ne_vals = np.meshgrid(-1.2 + (np.linspace(1, 60, 60) - 1) / 10 - 3,
                                   10 + 0.5 * (np.linspace(1, 15, 15) - 1) + 6)

    return Rbf(Te_vals, ne_vals, c31_data)
