function [psuccess st sst xvec yvec da a] = agentsearchrecovered(thresh, tau_e,tau_a,s,y_0,drw)
%Runs agent model search simulation using brightness levels of pixels of a
%movie as stimulus input. Agent navigates using response of to LIF neurons
%an evaluator (E), and an actuator (A).
%Input terms:
%   thresh - detection threshold of the evaluator LIF neuron
%   tau_e - decay constant (history) of the evaluator LIF neuron
%   tau_a - decay constnat (history) of the actuator LIF neuron
%   s - scale of amplitude decrease, controlled by actuator neuron
%   y_0 - initial y position of agent
%   drw - string input to draw all process inputs. Enter 'on' to draw.
%   Default 'off'.

%Suggested input values are: 
%   thresh = 10
%   tau_e = 24.65
%   tau_a = 92.21
%   s = 13.25

%Output terms:
%   psuccess - percent successful trials
%   st - average search time (length of search)
%   sst - average search time for successful trials
%   xvec - agent path, x dimension
%   yvec - agent path, y dimension
%   da - actuator input (evaluator spike train convolved with Gaussian)
%   a - actuator output (to check for egregious error)

if nargin == 5,
    drw = 'off';
end

npos = 1;%80;
nphase = 50;%10;
numit=npos*nphase;
i=1;
%f=1;
    
load(vids.mat);     %open video file
frames = vids(2).vid;   %use only the second video (it's longer). Comment out to use all videos.
clear vids
%Videos are stored as uint8 to use less RAM. This may have compatibility
%issues
f = 1;          

%for f=1:4,     %to cycle through videos
    
    %frames = vid(f).vid;
    fieldsize=size(frames); %use video infor to se tsizes
    Tspan=fieldsize(3);
    edges=fieldsize(1);

    xvec=nan(numit,Tspan);      %preallocate position vectors
    yvec=xvec;
    %rvec=nan(numit,1,length(delta_tau));
    rvec = nan(numit);
    avec=xvec;
    Ivec=xvec;
    evec=xvec;
    st=[];
    wcount=[];

    %Initial conditions
    x_0 = 650*ones(1,numit); 
    
    %Random X start position:
    %x_0 = round(50.*rand(1,numit)+645);
    
    %Random Y start position:
    %y_0 = round(80.*rand(1,numit)+176);
    
    %Center Y start position:
    if nargin < 5,
        y_0=176;
    end
    % you may have to adjust the number of inputs. deleting y_0 from input
    % may produce error.

    % rthresh=linspace(1,100,10);   %possible values for thresh
    % rtau_e=linspace(0.5,100,10);    %poss values for tau_e;
    % rtau_a=linspace(1,150,10);
    rs=linspace(10,50,20);           %...(1,50,10)
    rcmax=linspace(2,40,10);
    phases=linspace(0,2*pi,nphase);
%

%for f=1:length(rtau_e);        % for cycling through values of tau_e
    for pp = 1:npos;
        for qq = 1:nphase;
            
            %Preallocate variables
            i=nphase*(pp-1)+qq;
            r = zeros(1,Tspan);
            e= nan(1,Tspan);
            ytarg = r;              %center of y-oscillation
            a = r;
            da=r;
            t_s=r;

            %Initialize variables
            %     a_0 = 0;               %arbitrary, rescale as needed
            e_0 = 0;                %initial evaluator condition
            a_min = 0;              %initial actuator condition
            t = 1;

            %variables to augment -- RD, I will find you these values..
            %thresh = rthresh(4); %highest success
            dt = 1;
            %tau_e=rtau_e(9);
            %tau_a = 500;%rtau_a(9);              %amplitude timescale
            %s=rs(4); %rs(3);

            %initial conditions
            x = x_0(pp);
            y = y_0(pp);
            I = frames(y(1),x(1),1);

            ytarg = y;          %update with initial detection (ON response to plume)
            aim = y;
            tau_y = 15;          %drift should be integer, unless you round ytarg later
            e(1)=e_0;
            drft = 9;                        %windspeed/drag
            dytarg = 0;
            %dt_x=-11; %cos(dt_y);

            w_y= pi/15;%.06534*4;       %w = 13.4 deg/s (Lacey & Carde), .4467 deg/frm,.0077 rad/frm
            p_0=2*pi*phases(qq);     %initial phase (random)
            
            c_max=6.6; %rcmax(f); %6.6;
            c_min=c_max/5;
            %mean velocity should be about 5.68 pix/frame
            %period is roughly 0.1 periods/frame
            
            a_cap = fieldsize(1)/4;   %limits oscillation to half width of field
            %SZ-add conditional such that if amplitude is larger than a_cap,
            %it gets limited to a_cap
            
            a_max = (sqrt(2)/w_y)*sqrt((c_max+drft)^2-(c_min+drft)^2);  %constrains a_max such that it is not larger than c_0
            a(1) = 0;%a_max;

            le=15;
            
                for t = 1:Tspan%331,
                    %amp = a_max-a(t);
                    
                    if a(t)>a_max,      %enforces a_max amplitude limit
                        a(t)=a_max;
                    end
                                        
                    c(t)=-sqrt((-c_max-drft)^2-((a(t)^2)*((w_y)^2)/2))+drft;
                    
                    x(t+dt) = x(t)+(c(t))*dt;     %dt_x = abs(cos(dt_y*t));

                    %y(t+dt)=round((amp*sin(w_y*t))/v+ytarg(end));

                    y(t+dt) =(a_max-a(t))*sin(w_y*t+p_0)+ytarg(t);

                    %SPEED TESTER
                    % dy = y(t+dt)-y(t);
                    % spd(t)=sqrt((c(t))^2 + (dy)^2);
                    
                    %Boundary condition
                    if y(t+1)<1 || y(t+1)>=edges,
                        break
                    elseif y(t)<10 ||y(t)>edges-20,
                        break
                    elseif x(t)+c(t)<110,
                        x(t+1)=110;
                        y(t+1)=y(t);
                        break
                    end
                    
                    I(t+dt) = frames(round(y(t)),round(x(t)),t);

                    e(t+dt) = (1-dt/tau_e)*e(t) + I(t)*dt;
                    
                    aim(t+1)=aim(t);
                    
                    t_s(t)=t;      
                    
                    if e(t) >= thresh, 
                        w = ((e(t)-thresh)/thresh)*rand; %weights prob firing on salience
                        wcount=[wcount w];
                        if rand>=1-w,
                            r(t) = 1;    %"step" response, to alter a without a big headache
                            aim(t+1) = y(t);
                            if t+le>=Tspan,
                                le=Tspan-t;
                            end
                            da(t:t+le-1)=da(t:t+le-1)+(s/((le/10)*sqrt(2*pi))*exp(-((1:le)-le/2).^2/(2*(le/10))^2));
                    
                            %resets e
                            e(t+dt) = 0;%-10;
                        end                 
                    end
                                       
                    a(t+dt) = (1-dt/tau_a)*a(t) + da(t)*dt;     %Actuator LIF integration
                    
                    dytarg = (aim(t)-ytarg(t))/tau_y;
                    
                    ytarg(t+dt)= ytarg(t)+dytarg;

                    if a(t+dt)<a_min,
                        a(t+dt)= a_min;
                    end
                    
                end
                
                %Save to output vectors (for optimization)
                xvec(i,1:length(x),f)=x;
                yvec(i,1:length(y),f)=y;
                avec(i,1:length(a),f)=a;
                rvec(i,f)=sum(r);
                %Ivec(i,1:length(x))=I;
                %evec(i,1:length(x))=e; 
                
                if drw =='on',
                    figure
                    subplot(5,1,1)
                    plot(1:t,I);
                    ylabel('Pixel intensity')
                    title('Position-dependent stimulus')
                    
                    subplot(5,1,2)
                    plot(1:t,e,'k')
                    hold on, for i=1:t, plot(i,thresh,'r'), end
                    ylabel('Pixel intensity')
                    title('Evaluator-processed stimulus')
                    legend('Integrated stimulus','Detection threshold')
                    
                    subplot(5,1,3)
                    plot(1:t,'r')
                    ylabel('Spike raster')
                    title('Spike train of evaluator neuron')
                    
                    subplot(5,1,4)
                    plot(1:t,da);
                    title('Actuator stimulus (filtered evaluator output)')
                    
                    subplot(5,1,5)
                    plot(1:t,a)
                    title('Oscillation amplitude (Actuator output)')
                    ylabel('Amplitude (num pixels)')
                    
                end
                
                endpoint(i,1,f)=x(end);
                endpoint(i,2,f)=y(end);
                
                st(i,f)=t;
                
                if endpoint(i,2,f)>=200 && endpoint(i,2,f)<=230,
                    %wedge plumes endpoin bounds are 220,241
                    psuccess(i,f)=1;
                    succtime(i,f)=t;
                else
                    psuccess(i,f)=0;
                    succtime(i,f)=NaN;  %before, =331 
                end
                
                %psuccess=mean(psuccess);
        end
    end
%end
     
    % post hoc analysis
    sr(i,f)=sum(a==a_min)/t;    %calculates percent time surging (no osc)
    psuccess=mean(psuccess);    %percent successful trials
    sst=nanmean(succtime);      %average succesful duration of search
    st=mean(st);    %don't remember
    %st(f)=mean(st);
    
%end