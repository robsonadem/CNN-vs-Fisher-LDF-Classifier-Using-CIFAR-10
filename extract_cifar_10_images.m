load('test_batch.mat')
im=zeros(32,32,3);
for image=1:10000 
    
    R=data(image,1:1024);
    G=data(image,1025:2048);
    B=data(image,2049:3072);

    k=1;
    for j=1:32
      for i=1:32
          im(j,i,1)=R(k);
          im(j,i,2)=G(k);
          im(j,i,3)=B(k);
          k=k+1;
      end
    end  

     im=uint8(im);
     imwrite(im,strcat(int2str(image),'.png'),'png'); 
end