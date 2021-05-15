function Yhat = ENFM(Xtrn, Sz, m, lambda, th1,th2, sph, sop, wt, ot)
% ENFM is a neuro-fuzzy model based on adaptive Gath-Geva clustering.
% Xtrn: model input, Sz: labels, Yhat: model output
% m: fuzziness of clusters
% lambda: forgetting factor in RLS update rule
% th1: threshold for adding new clusters
% th2: threshold for merging the similar clusters
% wt: length of time-window (second)
% ot: length of overlap (second)

[m1, Q]=size(Xtrn);
Ytrn = -1.*ones(1,length(Sz));
R = eye(m1);
Fs = 256;   %sampling rate
N = (wt-ot)*Fs;      %length of residue

% This model is trained on-line based on the predefined SOP and SPH. 
% So, there is delay (pmax = sop + sph) between train sample and test sample
pmin = ceil(((sph*60*Fs)-(wt*Fs))/N);     % number of frames
pmax = ceil((((sph+sop)*60*Fs)-(wt*Fs))/N); 
p = pmax;

% Initialization
c=1;    % initial number of clusters
v(:,c) = Xtrn(:,1); 
A(:,:,c) = 70.*R; % initial covariance matrix
P(:,:,c) = 1e10.*R;
Ni(c,1) = 1;
w(:,c)=randn(m1,1); % initial weight matrix

for j=2+p:Q
           %% Test
        Ki = zeros(c,1);
        dm = zeros(c,1);
        u = zeros(c,1);
        phi = zeros(c,1);
        for i=1:c
            xv = Xtrn(:,j) - v(:,i);
            % Compute prior probability for each cluster
            Ki(i,1) = Ni(i,1)/sum(Ni,1);
            % Calculate distance of the new data point to each cluster
            d =(det(A(:,:,i))^(1/2))/Ki(i,1)*exp(1/2*xv'*(inv(A(:,:,i)))*xv);
            dm(i,1) = (d).^(-1/(m-1));
        end
        for i=1:c
            % Calculate membership of the new data point to each cluster
            u(i,1) = (dm(i,1) ./ sum(dm,1));
        end
        for i=1:c
            phi(i,1) = u(i,1)./sum(u,1);
            y(i,1) = w(:,i)'*Xtrn(:,j);
            Yi(i,1) = phi(i,1).*y(i,1);
        end
        Yhat(1,j) = sum(Yi,1);
        
    %% condition
    if Sz(1,j)==1
        Ytrn(1,j-pmax:j-pmin)=1;
    else
    end
    %% Train
        M = zeros(c,1);
        Ki = zeros(c,1);
        dm = zeros(c,1);
        u = zeros(c,1);
        phi = zeros(c,1);
        for i=1:c
            xv = Xtrn(:,j-p) - v(:,i);
            M(i,1) = exp(-1/2*xv'*inv(A(:,:,i))*xv);
            % Compute prior probability for each cluster
            Ki(i,1) = Ni(i,1)/sum(Ni,1);
            % Calculate distance of the new data point to each cluster
            d =(det(A(:,:,i))^(1/2))/Ki(i,1)*exp(1/2*xv'*(inv(A(:,:,i)))*xv);
            dm(i,1) = (d).^(-1/(m-1));
        end
        for i=1:c
            % Calculate membership of the new data point to each cluster
            u(i,1) = (dm(i,1) ./ sum(dm,1));
        end
        for i=1:c
            phi(i,1) = u(i,1)./sum(u,1);
        end
        if max(M(:,1)) < th1
            % Create new neuron
            c = c+1;
            v(:,c) = Xtrn(:,j-p);  %initial centers
            u(c,1) = 1;    %initial degree of membership 
            phi(c,1) = u(c,1)./sum(u(:,1),1);
            u(c,1) = phi(c,1);
            Ni(c,1) = mean(Ni(:,1)); 
            A(:,:,c) = mean(A,3);   % Calculate initial covariance matrix
            P(:,:,c) = 1e10.*R;
            w(:,c)=randn(m1,1); 
        else
        end
        % update parameters for each cluster
        for i=1:c
            um(i,1) = u(i,1).^m;
            Ni1 = Ni(i,1) + um(i,1);
            xvhat = Xtrn(:,j-p) - v(:,i);
            v(:,i) = v(:,i) + (xvhat./Ni1).*um(i,1);
            A(:,:,i) = (Ni(i,1)./Ni1).*(A(:,:,i)+(um(i,1)./Ni1)*xvhat*(xvhat'));
            Ni(i,1) = Ni1;
            K(:,i) = (P(:,:,i)*Xtrn(:,j-p))/(Xtrn(:,j-p)'*P(:,:,i)*Xtrn(:,j-p)+lambda/phi(i,1));
            E1 = Ytrn(1,j-p)-w(:,i)'*Xtrn(:,j-p);
            w(:,i) = w(:,i)+K(:,i)*E1;
            E2 = Ytrn(1,j-p)-w(:,i)'*Xtrn(:,j-p);      %regression error
            P(:,:,i) = (1/lambda)*(R-K(:,i)*Xtrn(:,j-p)')*P(:,:,i);
        end

        while c~=1
            cb = combntns(1:c,2);
            [r,~]=size(cb);
            s=zeros(r,2);
            c1=c;
            for i=1:r
                s(i,1) = exp(-1/2*(v(:,cb(i,2))-v(:,cb(i,1)))'*inv(A(:,:,cb(i,1)))*(v(:,cb(i,2))-v(:,cb(i,1))));
                s(i,2) = exp(-1/2*(v(:,cb(i,1))-v(:,cb(i,2)))'*inv(A(:,:,cb(i,2)))*(v(:,cb(i,1))-v(:,cb(i,2))));
                if s(i,1)>th2 && s(i,2)>th2
                    % merge two neurons
                    c=c-1;
                    Nk = sum(Ni([cb(i,1) cb(i,2)],1));
                    vk = (Ni(cb(i,1),1).*v(:,cb(i,1))+Ni(cb(i,2),1).*v(:,cb(i,2)))./Nk;
                    Nm2Nn = (Ni(cb(i,1),1).^2)*Ni(cb(i,2),1);
                    NmNn2 = Ni(cb(i,1),1).*(Ni(cb(i,2),1).^2);
                    Ak = 1/(Nk.^3).*((Ni(cb(i,1),1).^3+2.*Nm2Nn+NmNn2).*A(:,:,cb(i,1))...
                    +(Ni(cb(i,2),1).^3+2.*NmNn2+Nm2Nn).*A(:,:,cb(i,2))+(NmNn2+Nm2Nn)...
                    .*(v(:,cb(i,2))-v(:,cb(i,1)))*(v(:,cb(i,2))-v(:,cb(i,1)))');
                    Ni(cb(i,2),:) = [];
                    Ni(cb(i,1),:) = Nk;
                    v(:,cb(i,2)) = [];
                    v(:,cb(i,1)) = vk;
                    A(:,:,cb(i,2)) = [];
                    A(:,:,cb(i,1)) = Ak;
                    break
                else
                end
            end
            if c==c1
                break
            end
        end
   C(j,1)=c;
end