clear;

r=0;                            %tone of rest/zero vector amplitude 
t=0.342;                        %the base length of a note in seconds

%% Notes list
a=440*(2^(0/12));
b=440*(2^(2/12));
c=440*(2^(3/12));
d=440*(2^(5/12));
e=440*(2^(7/12));
f=440*(2^(9/12));               %actually f sharp
g=440*(2^(10/12));

%% Duration list
fs= 12000;                      %sampling frequency
en=[0:(1/fs):t/2-(1/fs)];       %eigth note
q=[0:(1/fs):t*0.98-(1/fs)];     %quarter note
fer=[0:(1/fs):t*(4/3)-(1/fs)];  %fermata
h=[0:(1/fs):t*2-(1/fs)];        %half note
w=[0:(1/fs):t*4-(1/fs)]         %whole note
rt=[0:(1/fs):0.007-(1/fs)];     %shortrest length

%% Anonymous function to generate tone and length
t = @(note, duration) cos(2*pi*note*duration);


%% Right hand vectors
y1=[0.5*(t(d*2, en)+t(b*2, en)) 0.5*t(g, en) 0.5*(t(d*2, h)+t(b*2, h)) t(r, rt) 0.5*(t(d*2, en)+t(b*2, en)) 0.5*(t(g, en))...
    0.4*(t(d*2, en)+t(b*2, en)) 0.4*t(f, en) 0.4*(t(d*2, h)+t(b*2, h)) t(r, rt) 0.4*(t(d*2, en)+t(b*2, en)) 0.4*t(f, en)...
    0.4*(t(e*2, en)+t(b*2, en)) 0.4*t(g, en) 0.4*(t(e*2, h)+t(b*2, h)) t(r, rt) 0.4*(t(e*2, en)+t(b*2, en)) 0.4*t(g, en)...
    0.5*(t(e*2, q)+t(c*2, q)) t(r, rt) 0.5*(t(e*2, q)+t(c*2, q)) t(r, rt) 0.5*(t(e*2, en)+t(c*2, en)) 0.5*t(b*2, en) 0.5*t(a*2, en) 0.5*t(b*2, en)...
    0.5*(t(d*2, en)+t(b*2, en)) 0.5*t(g, en) 0.5*(t(d*2, h)+t(b*2, h)) t(r, rt) 0.5*(t(d*2, en)+t(b*2, en)) 0.5*t(g, en)...
    0.6*(t(d*2, en)+t(b*2, en)) 0.6*t(f, en) 0.6*(t(d*2, h)+t(b*2, h)) t(r, rt) 0.6*(t(d*2, en)+t(b*2, en)) 0.6*t(f, en)...
    0.6*(t(e*2, en)+t(b*2, en)) 0.6*t(g, en) 0.6*(t(e*2, h)+t(b*2, h)) t(r, rt) 0.6*(t(e*2, en)+t(b*2, en)) 0.6*t(g, en)...
    0.7*(t(e*2, q)+t(c*2, q)) t(r, rt) 0.7*(t(e*2, q)+t(c*2, q)) t(r, rt) 0.7*(t(e*2, en)+t(c*2, en)) 0.7*t(b*2, en) 0.7*t(a*2, en) 0.7*t(b*2, en)...
    t(g, h) t(g*2, h)...
    t(r, h) t(f*2, en) t(g*2, en) t(a*4, en) t(f*2, en)...
    t(e*2, h) t(r, rt) t(e*2, q) t(r, rt) t(e*2, q)...
    t(c*2, q) t(r, rt) t(c*2, q) t(r, rt) t(c*2, en) t(b*2, en) t(a*2, en) t(b*2, en)...
    t(g, h)+t(d, h) t(g*2, h)+t(d*2, h)...
    t(r, h) t(f*2, en) t(g*2, en) t(a*4, en) t(f*2, en)...
    t(e, h)+t(e*2, h) t(r, rt) t(e, q)+t(e*2, q) t(r, rt) t(e, q)+t(e*2, q)...
    t(c, q)+t(c*2, q) t(r, rt) t(c, q)+t(c*2, q) t(r, rt) t(c*2, en) t(b*2, en) t(a*2, en) t(b*2, en)...
    ];
    
    
y2= [t(b*2, en)+t(d*2, en) t(g, en) t(b*2, h)+t(d*2, h) t(b*2, en)+t(d*2, en) t(g, en) t(b*2, en)+t(d*2, en) t(f, en) t(b*2, h)+t(d*2, h)... 
    t(b*2, en)+t(d*2, en) t(f, en) t(b*2, en)+t(e*2, en) t(g, en) t(b*2, h)+t(e*2, h) t(b*2, en)+t(e*2, en) t(g, en)...
    t(c*2, q)+t(e*2, q) t(c*2, q)+t(e*2, q) t(c*2, en)+t(e*2, en) t(b*2, en) t(a*2, en) t(b*2, h) t(g, en)...
    zeros(1, 166)
    ];
    


y3= [t(r, rt) t(g, en) t(d, en) t(g, en) t(b*2, en) t(g, en) t(b*2, en) t(d*2, en) t(b*2, en) t(d*2, en) t(g*2, h)+t(g, h)
    ];


%% Left hand vectors
y4=[0.5*t(g, w)+0.3*t(g/2, w)...
    0.5*t(b, w)+0.3*t(b/2, w)...
    0.5*t(e, w)+0.3*t(e/2, w)...
    0.6*(t(c, h)+t(c/2, h)) t(r, rt) 0.5*(t(c, h)+t(c/2, h))...
    0.7*(t(g, q)+t(g/2, q)) t(r, rt) 0.7*(t(g, q)+t(g/2, q)) t(r, rt) 0.7*(t(g, q)+t(g/2, q)) t(r, rt) 0.7*(t(g, q)+t(g/2, q))...
    t(b, q)+t(b/2, q) t(r, rt) t(b, q)+t(b/2, q) t(r, rt) t(b, q)+t(b/2, q) t(r, rt) t(b, q)+t(b/2, q)...
    t(e, q)+t(e/2, q) t(r, rt) t(e, q)+t(e/2, q) t(r, rt) t(e, q)+t(e/2, q) t(r, rt) t(e, q)+t(e/2, q)...
    t(c/2, en)+t(c, en) t(r, rt) t(c/2, en)+t(c, en) t(r, rt) t(c/2, en)+t(c, en) t(r, rt) t(c/2, en)+t(c, en) t(r, rt) t(c/2, en)+t(c, en) t(r, rt) t(c/2, en)+t(c, en) t(r, rt) t(c/2, en)+t(c, en) t(r, rt) t(c/2, en)+t(c, en)...
    t(g/4, en) t(d/2, en) t(g/2, en) t(d/2, en) t(g/4, en) t(d/2, en) t(g/2, en) t(d/2, en)...
    t(b/4, en) t(f/4, en) t(b/2, en) t(f/4, en) t(b/4, en) t(f/4, en) t(b/2, en) t(f/4, en)...
    t(e/4, en) t(b/2, en) t(e/2, en) t(b/2, en) t(e/4, en) t(b/2, en) t(e/2, en) t(b/2, en)...
    t(c/4, en) t(g/4, en) t(c/2, en) t(g/4, en) t(c/2, en) t(g/4, en) t(c, en) t(g/4, en)...
    t(g/4, en) t(d/2, en) t(g/2, en) t(d/2, en) t(g/4, en) t(d/2, en) t(g/2, en) t(d/2, en)...
    t(b/2, en) t(f/2, en) t(b, en) t(f/2, en) t(b/2, en) t(f/2, en) t(b, en) t(f/2, en)...
    t(e/2, en) t(b/2, en) t(e, en) t(b/2, en) t(e/2, en) t(b/2, en) t(e, en) t(b/2, en)...
    t(c/4, en) t(g/4, en) t(c/2, en) t(g/4, en) t(c/2, en) t(g/4, en) t(c, en) t(g/4, en)...
    zeros(1, 84)
    ];
    
y5= [t(g/4, w)+t(2, w) t(b/2, h)+t(b, h)...
    t(r, w) t(r, h)...
    t(r, w) t(r, h)
    ];
    

y6= [t(g/2, q) t(b, q) t(d, q) t(g, q)...
    zeros(1, 10676)
    ];

s1 = size(y2);
s2 = size(y5);
s3 = size(y3);
s4 = size(y6);

%% decrescendo window function
x1=linspace(0,1,s1(2));
x2=linspace(0,1,s2(2));
d1=exp(-x1./0.2)+0.2;
d2=exp(-x2./0.02)+0.1;


%% crescendo window function
x3=linspace(0, 1, s3(2));
x4=linspace(0, 1, s4(2));
c1=x3.^1.7+0.2;
c2=x4.^2.2;

%% Combining all vectors to single vector
y=[y1+(0.5*y4), (d1.*y2)+(d2.*y5), (c1.*y3)+(c2.*y6)];
y=y/(max(y));
soundsc(y, fs)

%% Writing to .wav file
filename = 'zhang_theoffice.wav';
audiowrite(filename,y,fs);

%% Plots and analysis

clf;
figure(1)
plot(x1, d1)
hold
title('Decrescendo window function')

figure(2)
plot(x3, c1)
hold
title('Crescendo window function')