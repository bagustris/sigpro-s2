plt.figure(figsize = (15,10))
for i in range(p):
    Y[:,:,i] = perform_stft(y[:,i],w,q,n)
    plt.subplot(3,1, i+1)
    plot_spectrogram(Y[:,:,i],"Microphone #%i" %(i+1))

# plt.show()