ff=@streamsearch;

%1)thresh, 2)tau_e, 3)tau_a, 4)s

thresh = linspace(2,25,5);
s = linspace(1,50,5);
tau_e = linspace(1,25,25);
tau_a = linspace(1,50,25);
ypos = linspace(176,256,3);

global vids
load vids.mat

%a= length(s);
filename = ['run',date];
tic

for k=1:3,
for m = 1:5,
    for n = 1:5,
    for l = 1:25,
        for p = 1:24,
             [ps(:,k,l,p,n,m), sst(:,k,l,p,n,m)]=feval(ff,thresh(n),tau_e(p),tau_a(l),s(m),ypos(k)); 
        end
    end    
    end
   
end
    if k==1,
        '50% done'
        %toc
    elseif k==2
        'Almost done!'
        toc
    end
    
    save(filename,'sst','ps','tau_a','tau_e','thresh','ypos','s')
    
end
        
%index result vectors; use sort command to order max values. find index
%with highest ps and sst.




