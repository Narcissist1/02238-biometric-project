function Compute_and_Plot_DET(scoresMated, scoresNonMated)[P_miss,P_fa] = Compute_DET(scoresMated, scoresNonMated);%Plot_DET (P_miss,P_fa, 'algorithm', 'b'); hold onPlot_DET (P_miss, P_fa, 'identification', 'b', 1);legend('example')set(gca,'FontSize',20)endfunction [Pmiss, Pfa] = Compute_DET(true_scores, false_scores)num_true = max(size(true_scores));num_false = max(size(false_scores));total=num_true+num_false;Pmiss = zeros(num_true+num_false+1, 1);Pfa   = zeros(num_true+num_false+1, 1);scores(1:num_false,1) = false_scores;scores(1:num_false,2) = 0;scores(num_false+1:total,1) = true_scores;scores(num_false+1:total,2) = 1;scores=DETsort(scores);sumtrue=cumsum(scores(:,2),1);sumfalse=num_false - ([1:total]'-sumtrue);Pmiss(1) = 0;Pfa(1) = 1.0;Pmiss(2:total+1) = sumtrue  ./ num_true;Pfa(2:total+1)   = sumfalse ./ num_false;endfunction [y,ndx] = DETsort(x, col)if nargin<1, error('Not enough input arguments.'); endif ~ismatrix(x), error('X must be a 2-D matrix.'); endif nargin<2, col = 1:size(x,2); endif isempty(x), y = x; ndx = []; return, endndx = (1:size(x,1))';[v,ind] = sort(x(ndx,2));ndx = ndx(ind);ndx(1:size(x,1)) = ndx(size(x,1):-1:1);[v,ind] = sort(x(ndx,1));ndx = ndx(ind);y = x(ndx,:);endfunction h = Plot_DET (Pmiss, Pfa, opt_type, plot_code, abbreviate_axes, opt_title, opt_thickness)Npts = max(size(Pmiss));if Npts ~= max(size(Pfa))    error ('vector size of Pmiss and Pfa not equal in call to Plot_DET');endif ~exist('plot_code', 'var')    plot_code = 'b';endif ~exist('opt_thickness', 'var')    opt_thickness = 3;endif ~exist('opt_type', 'var')    opt_type = 'other';endif ~exist('abbreviate_axes', 'var')    abbreviate_axes = false;endif ~exist('opt_tite', 'var')    opt_title = 'DET Curves';endSet_DET_limits;h = thick(opt_thickness,plot(ppndf(Pfa), ppndf(Pmiss), plot_code));Make_DET(opt_type, abbreviate_axes, opt_title);endfunction Make_DET(opt_type, abbreviate_axes, opt_title)pticks = [0.00001 0.00002 0.00005 0.0001  0.0002   0.0005 ...    0.001   0.002   0.005   0.01    0.02     0.05 ...    0.1     0.2     0.4     0.6     0.8      0.9 ...    0.95    0.98    0.99    0.995   0.998    0.999 ...    0.9995  0.9998  0.9999  0.99995 0.99998  0.99999];xlabels = [' 0.001' ; ' 0.002' ; ' 0.005' ; ' 0.01 ' ; ' 0.02 ' ; ' 0.05 ' ; ...    '  0.1 ' ; '  0.2 ' ; ' 0.5  ' ; '  1   ' ; '  2   ' ; '  5   ' ; ...    '  10  ' ; '  20  ' ; '  40  ' ; '  60  ' ; '  80  ' ; '  90  ' ; ...    '  95  ' ; '  98  ' ; '  99  ' ; ' 99.5 ' ; ' 99.8 ' ; ' 99.9 ' ; ...    ' 99.95' ; ' 99.98' ; ' 99.99' ; '99.995' ; '99.998' ; '99.999'];ylabels = xlabels;global DET_limits;if isempty(DET_limits)    Set_DET_limits;endPmiss_min = DET_limits(1);Pmiss_max = DET_limits(2);Pfa_min   = DET_limits(3);Pfa_max   = DET_limits(4);ntick = max(size(pticks));for n=ntick:-1:1    if (Pmiss_min <= pticks(n))        tmin_miss = n;    end    if (Pfa_min <= pticks(n))        tmin_fa = n;    endendfor n=1:ntick    if (pticks(n) <= Pmiss_max)        tmax_miss = n;    end    if (pticks(n) <= Pfa_max)        tmax_fa = n;    endendset (gca, 'xlim', ppndf([Pfa_min Pfa_max]));set (gca, 'xtick', ppndf(pticks(tmin_fa:tmax_fa)));set (gca, 'xticklabel', xlabels(tmin_fa:tmax_fa,:));set (gca, 'xgrid', 'on');set (gca, 'ylim', ppndf([Pmiss_min Pmiss_max]));set (gca, 'ytick', ppndf(pticks(tmin_miss:tmax_miss)));set (gca, 'yticklabel', ylabels(tmin_miss:tmax_miss,:));set (gca, 'ygrid', 'on')if strcmp(opt_type,'algorithm')    if abbreviate_axes        xlabel = 'FMR (in %)';        ylabel = 'FNMR (in %)';    else        xlabel = 'False Match Rate (in %)';        ylabel = 'False Non-Match Rate (in %)';    endelseif strcmp(opt_type,'system')    if abbreviate_axes        xlabel = 'FAR (in %)';        ylabel = 'FRR (in %)';    else        xlabel = 'False Acceptance Rate (in %)';        ylabel = 'False Rejection Rate (in %)';    endelseif strcmp(opt_type,'PAD')    if abbreviate_axes        xlabel = 'APCER (in %)';        ylabel = 'BPCER (in %)';    else        xlabel = 'Attack Presentation Classification Error Rate (in %)';        ylabel = 'Bona Fide Presentation Classification Error Rate (in %)';    endelseif strcmp(opt_type,'identification')    if abbreviate_axes        xlabel = 'FPIR (in %)';        ylabel = 'FNIR (in %)';    else        xlabel = 'False Positive Identification Rate (in %)';        ylabel = 'False Negative Identification Rate (in %)';    endelse    xlabel = 'Type I Error Rate (in %)';    ylabel = 'Type II Error Rate (in %)';endset(set(get(gca, 'XLabel'), 'String', xlabel))set(set(get(gca, 'YLabel'), 'String', ylabel))title(opt_title)set (gca, 'box', 'on');axis('square');axis(axis);endfunction Set_DET_limits(Pmiss_min, Pmiss_max, Pfa_min, Pfa_max)Pmiss_min_default = 0.0005+eps;Pmiss_max_default = 0.5-eps;Pfa_min_default = 0.0005+eps;Pfa_max_default = 0.5-eps;global DET_limits;if (~isempty(DET_limits))    Pmiss_min_default = DET_limits(1);    Pmiss_max_default = DET_limits(2);    Pfa_min_default  = DET_limits(3);    Pfa_max_default  = DET_limits(4);endif ~(exist('Pmiss_min', 'var'))     Pmiss_min = Pmiss_min_default;endif ~(exist('Pmiss_max', 'var'))     Pmiss_max = Pmiss_max_default;endif ~(exist('Pfa_min', 'var'))     Pfa_min = Pfa_min_default;endif ~(exist('Pfa_max', 'var'))      Pfa_max = Pfa_max_default;endPmiss_min = max(Pmiss_min,eps);Pmiss_max = min(Pmiss_max,1-eps);if Pmiss_max <= Pmiss_min    Pmiss_min = eps;    Pmiss_max = 1-eps;endPfa_min = max(Pfa_min,eps);Pfa_max = min(Pfa_max,1-eps);if Pfa_max <= Pfa_min    Pfa_min = eps;    Pfa_max = 1-eps;endDET_limits = [Pmiss_min Pmiss_max Pfa_min Pfa_max];endfunction norm_dev = ppndf (cum_prob)SPLIT =  0.42;A0 =   2.5066282388;A1 = -18.6150006252;A2 =  41.3911977353;A3 = -25.4410604963;B1 =  -8.4735109309;B2 =  23.0833674374;B3 = -21.0622410182;B4 =   3.1308290983;C0 =  -2.7871893113;C1 =  -2.2979647913;C2 =   4.8501412713;C3 =   2.3212127685;D1 =   3.5438892476;D2 =   1.6370678189;[Nrows, Ncols] = size(cum_prob);norm_dev = zeros(Nrows, Ncols);cum_prob(cum_prob>= 1.0) = 1-eps;cum_prob(cum_prob<= 0.0) = eps;R = zeros(Nrows, Ncols);adj_prob=cum_prob-0.5;centerindexes = find(abs(adj_prob) <= SPLIT);tailindexes   = find(abs(adj_prob) > SPLIT);R(centerindexes) = adj_prob(centerindexes) .* adj_prob(centerindexes);norm_dev(centerindexes) = adj_prob(centerindexes) .* ...    (((A3 .* R(centerindexes) + A2) .* R(centerindexes) + A1) .* R(centerindexes) + A0);norm_dev(centerindexes) = norm_dev(centerindexes) ./ ((((B4 .* R(centerindexes) + B3) .* R(centerindexes) + B2) .* ...    R(centerindexes) + B1) .* R(centerindexes) + 1.0);right = find(cum_prob(tailindexes)> 0.5);left  = find(cum_prob(tailindexes)< 0.5);R(tailindexes) = cum_prob(tailindexes);R(tailindexes(right)) = 1 - cum_prob(tailindexes(right));R(tailindexes) = sqrt ((-1.0) .* log (R(tailindexes)));norm_dev(tailindexes) = (((C3 .* R(tailindexes) + C2) .* R(tailindexes) + C1) .* R(tailindexes) + C0);norm_dev(tailindexes) = norm_dev(tailindexes) ./ ((D2 .* R(tailindexes) + D1) .* R(tailindexes) + 1.0);norm_dev(tailindexes(left)) = norm_dev(tailindexes(left)) .* -1.0;endfunction [lh] = thick(w,lh)for i=1:length(lh)    set (lh(i),'LineWidth',w);endend