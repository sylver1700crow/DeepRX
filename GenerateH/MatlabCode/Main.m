clc
clear
close all

simParameters = struct();       % Clear simParameters variable to contain all key simulation parameters 所有参数放在一起
simParameters.NFrames = 2;      % Number of 10 ms frames 帧个数 一个帧10ms 10个子帧 20个slot 一个slot 7个symbol 
simParameters.SNRIn = 30; % SNR range (dB)
simParameters.PerfectChannelEstimator = true; %完美信道估计
simParameters.DisplayDiagnostics = false;
Nt=1;
Nr=2;
numslot=100;
S=12; % OFDM Symbols 
F=24; % Subcarriers per OFDM symbol
Ideal_H = zeros(int8(numslot*600/F),S,F,Nr,2); %理想信道 20000 *12
int8(numslot*600/F)

Hls = zeros(int8(numslot*600/F),S,F,Nr,2); %理想信道 20000 *12
Received_Y= zeros(int8(numslot*600/F),S,F,Nr,2); 
Ideal_X=zeros(int8(numslot*600/F),S,F,2);
Transmit_X=zeros(int8(numslot*600/F),S,F,2);
simParameters.Carrier = nrCarrierConfig;         %RB=12子载波 7个symbol 即一个slot
simParameters.Carrier.NSizeGrid = 51;            % Bandwidth in number of resource blocks (51 RBs at 30 kHz SCS for 20 MHz BW) 有51个RB 即612个子载波
simParameters.Carrier.SubcarrierSpacing = 30;    % 15, 30, 60, 120, 240 (kHz)
simParameters.Carrier.CyclicPrefix = 'Normal';   % 'Normal' or 'Extended' (Extended CP is relevant for 60 kHz SCS only)
simParameters.Carrier.NCellID = 1;               % Cell identity
MaximunDoppler=75;                              %[15,45,75,105,135,165]


% SS burst configuration 
% The burst can be disabled by setting the SSBTransmitted field to all zeros
simParameters.SSBurst = struct(); %ssb是同步信号
simParameters.SSBurst.BlockPattern = 'Case B';    % 30 kHz subcarrier spacing
simParameters.SSBurst.SSBTransmitted = [0 1 0 1]; % Bitmap indicating blocks transmitted in the burst
simParameters.SSBurst.SSBPeriodicity = 20;        % SS burst set periodicity in ms (5, 10, 20, 40, 80, 160)

% PDSCH/DL-SCH parameters 系统基本设置
simParameters.PDSCH = nrPDSCHConfig;      % This PDSCH definition is the basis for all PDSCH transmissions in the BLER simulation
simParameters.PDSCHExtension = struct();  % This structure is to hold additional simulation parameters for the DL-SCH and PDSCH

% Define PDSCH time-frequency resource allocation per slot to be full grid
% (single full grid BWP) 把RB映射成PRB 作为下行信号
simParameters.PDSCH.PRBSet = 0:simParameters.Carrier.NSizeGrid-1;                 % PDSCH PRB allocation
simParameters.PDSCH.SymbolAllocation = [0,simParameters.Carrier.SymbolsPerSlot];  % Starting symbol and number of symbols of each PDSCH allocation
simParameters.PDSCH.MappingType = 'A';     % PDSCH mapping type ('A'(slot-wise),'B'(non slot-wise))

% Scrambling identifiers
simParameters.PDSCH.NID = simParameters.Carrier.NCellID;
simParameters.PDSCH.RNTI = 1;

% PDSCH resource block mapping (TS 38.211 Section 7.3.1.6)
simParameters.PDSCH.VRBToPRBInterleaving = 0; % Disable interleaved resource mapping
simParameters.PDSCH.VRBBundleSize = 4;

% Define the number of transmission layers to be used
simParameters.PDSCH.NumLayers = 1;            % Number of PDSCH transmission layers

% Define codeword modulation and target coding rate 调制方式
% The number of codewords is directly dependent on the number of layers so ensure that 
% layers are set first before getting the codeword number
if simParameters.PDSCH.NumCodewords > 1                             % Multicodeword transmission (when number of layers being > 4)
    simParameters.PDSCH.Modulation = {'16QAM','16QAM'};             % 'QPSK', '16QAM', '64QAM', '256QAM'
    simParameters.PDSCHExtension.TargetCodeRate = [490 490]/1024;   % Code rate used to calculate transport block sizes
else
    simParameters.PDSCH.Modulation = 'QPSK';                       % 'QPSK', '16QAM', '64QAM', '256QAM'
    simParameters.PDSCHExtension.TargetCodeRate = 490/1024;         % Code rate used to calculate transport block sizes
end

% DM-RS and antenna port configuration (TS 38.211 Section 7.4.1.1)
simParameters.PDSCH.DMRS.DMRSPortSet = 0:simParameters.PDSCH.NumLayers-1; % DM-RS ports to use for the layers
simParameters.PDSCH.DMRS.DMRSTypeAPosition = 2;      % Mapping type A only. First DM-RS symbol position (2,3)
simParameters.PDSCH.DMRS.DMRSLength = 1;             % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
simParameters.PDSCH.DMRS.DMRSAdditionalPosition = 0; % Additional DM-RS symbol positions (max range 0...3)
simParameters.PDSCH.DMRS.DMRSConfigurationType = 2;  % DM-RS configuration type (1,2)
simParameters.PDSCH.DMRS.NumCDMGroupsWithoutData = 1;% Number of CDM groups without data
simParameters.PDSCH.DMRS.NIDNSCID = 1;               % Scrambling identity (0...65535)
simParameters.PDSCH.DMRS.NSCID = 0;                  % Scrambling initialization (0,1)

% PT-RS configuration (TS 38.211 Section 7.4.1.2)
simParameters.PDSCH.EnablePTRS = 0;                  % Enable or disable PT-RS (1 or 0)
simParameters.PDSCH.PTRS.TimeDensity = 1;            % PT-RS time density (L_PT-RS) (1, 2, 4)
simParameters.PDSCH.PTRS.FrequencyDensity = 2;       % PT-RS frequency density (K_PT-RS) (2 or 4)
simParameters.PDSCH.PTRS.REOffset = '00';            % PT-RS resource element offset ('00', '01', '10', '11')
simParameters.PDSCH.PTRS.PTRSPortSet = [];           % PT-RS antenna port, subset of DM-RS port set. Empty corresponds to lower DM-RS port number

% Reserved PRB patterns, if required (for CORESETs, forward compatibility etc)
simParameters.PDSCH.ReservedPRB{1}.SymbolSet = [];   % Reserved PDSCH symbols
simParameters.PDSCH.ReservedPRB{1}.PRBSet = [];      % Reserved PDSCH PRBs
simParameters.PDSCH.ReservedPRB{1}.Period = [];      % Periodicity of reserved resources

% Additional simulation and DL-SCH related parameters
%
% PDSCH PRB bundling (TS 38.214 Section 5.1.2.3)
simParameters.PDSCHExtension.PRGBundleSize = [];      % 2, 4, or [] to signify "wideband"
% HARQ process and rate matching/TBS parameters
simParameters.PDSCHExtension.XOverhead = 6*simParameters.PDSCH.EnablePTRS; % Set PDSCH rate matching overhead for TBS (Xoh) to 6 when PT-RS is enabled, otherwise 0
simParameters.PDSCHExtension.NHARQProcesses = 16;     % Number of parallel HARQ processes to use
simParameters.PDSCHExtension.EnableHARQ = true;       % Enable retransmissions for each process, using RV sequence [0,2,3,1]

% LDPC decoder parameters
% Available algorithms: 'Belief propagation', 'Layered belief propagation', 'Normalized min-sum', 'Offset min-sum'
simParameters.PDSCHExtension.LDPCDecodingAlgorithm = 'Layered belief propagation';
simParameters.PDSCHExtension.MaximumLDPCIterationCount = 6;

% Define the overall transmission antenna geometry at end-points
% If using a CDL propagation channel then the integer number of antenna elements is
% turned into an antenna panel configured when the channel model object is created
simParameters.NTxAnts = Nt;                        % 发端天线数Number of PDSCH transmission antennas (1,2,4,8,16,32,64,128,256,512,1024) >= NumLayers
if simParameters.PDSCH.NumCodewords > 1           % Multi-codeword transmission
    simParameters.NRxAnts = Nr;                    % 收端天线数目 Number of UE receive antennas (even number >= NumLayers)
else
    simParameters.NRxAnts = Nr;                    % Number of UE receive antennas (1 or even number >= NumLayers)
end

% Define the general CDL/TDL propagation channel parameters
simParameters.DelayProfile = 'CDL-A';   % Use CDL-C model (Urban macrocell model)
simParameters.DelaySpread = 30e-9;
simParameters.DopplerIn = [15,45,75,105,135,165];
simParameters.MaximumDopplerShift = MaximunDoppler; 

% Cross-check the PDSCH layering against the channel geometry 
validateNumLayers(simParameters);
waveformInfo = nrOFDMInfo(simParameters.Carrier); % Get information about the baseband waveform after OFDM modulation step
% Constructed the CDL or TDL channel model object
if contains(simParameters.DelayProfile,'CDL','IgnoreCase',true)   
    channel = nrCDLChannel; % CDL channel object   
    % Turn the overall number of antennas into a specific antenna panel
    % array geometry. The number of antennas configured is updated when
    % nTxAnts is not one of (1,2,4,8,16,32,64,128,256,512,1024) or nRxAnts
    % is not 1 or even.
    [channel.TransmitAntennaArray.Size, channel.ReceiveAntennaArray.Size] = ...
        hArrayGeometry(simParameters.NTxAnts,simParameters.NRxAnts);
    nTxAnts = prod(channel.TransmitAntennaArray.Size);
    nRxAnts = prod(channel.ReceiveAntennaArray.Size);
    simParameters.NTxAnts = nTxAnts;
    simParameters.NRxAnts = nRxAnts;
else
    channel = nrTDLChannel; % TDL channel object  
    % Set the channel geometry
    channel.NumTransmitAntennas = simParameters.NTxAnts;
    channel.NumReceiveAntennas = simParameters.NRxAnts;
end
% Assign simulation channel parameters and waveform sample rate to the object
channel.DelayProfile = simParameters.DelayProfile;
channel.DelaySpread = simParameters.DelaySpread;
channel.MaximumDopplerShift = simParameters.MaximumDopplerShift;
channel.SampleRate = waveformInfo.SampleRate;
chInfo = info(channel);
maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) + chInfo.ChannelFilterDelay;
simParameters.SSBurst.NCellID = simParameters.Carrier.NCellID;        
simParameters.SSBurst.SampleRate = waveformInfo.SampleRate;
ssbInfo = hSSBurstInfo(simParameters.SSBurst);
[mappedPRB,mappedSymbols] = mapNumerology(ssbInfo.OccupiedSubcarriers,ssbInfo.OccupiedSymbols,ssbInfo.NRB,simParameters.Carrier.NSizeGrid,ssbInfo.SubcarrierSpacing,simParameters.Carrier.SubcarrierSpacing);
% Configure the PDSCH to reserve these resources so that the PDSCH
% transmission does not overlap the SS burst
reservation = nrPDSCHReservedConfig;
reservation.SymbolSet = mappedSymbols;
reservation.PRBSet = mappedPRB;
reservation.Period = simParameters.SSBurst.SSBPeriodicity * (simParameters.Carrier.SubcarrierSpacing/15); % Period in slots
simParameters.PDSCH.ReservedPRB{end+1} = reservation;

%% Processing Loop
% Array to store the maximum throughput for all SNR points
maxThroughput = zeros(length(simParameters.SNRIn),1); 
% Array to store the simulation throughput for all SNR points
simThroughput = zeros(length(simParameters.SNRIn),1);

% Set up Redundancy Version (RV) sequence to be used, according to the HARQ configuration 
if simParameters.PDSCHExtension.EnableHARQ
    % In the final report of RAN WG1 meeting #91 (R1-1719301), it was
    % observed in R1-1717405 that if performance is the priority, [0 2 3 1]
    % should be used. If self-decodability is the priority, it should be
    % taken into account that the upper limit of the code rate at which
    % each RV is self-decodable is in the following order: 0>3>2>1
    rvSeq = [0 2 3 1];
else
    % HARQ disabled - single transmission with RV=0, no retransmissions
    rvSeq = 0; 
end

% Create DL-SCH encoder system object to perform transport channel encoding
encodeDLSCH = nrDLSCH;
encodeDLSCH.MultipleHARQProcesses = true;
encodeDLSCH.TargetCodeRate = simParameters.PDSCHExtension.TargetCodeRate;

% Create DL-SCH decoder system object to perform transport channel decoding
% Use layered belief propagation for LDPC decoding, with half the number of
% iterations as compared to the default for belief propagation decoding
decodeDLSCH = nrDLSCHDecoder;
decodeDLSCH.MultipleHARQProcesses = true;
decodeDLSCH.TargetCodeRate = simParameters.PDSCHExtension.TargetCodeRate;
decodeDLSCH.LDPCDecodingAlgorithm = simParameters.PDSCHExtension.LDPCDecodingAlgorithm;
decodeDLSCH.MaximumLDPCIterationCount = simParameters.PDSCHExtension.MaximumLDPCIterationCount;

for snrIdx = 1:numel(simParameters.SNRIn)      % comment out for parallel computing
% parfor snrIdx = 1:numel(simParameters.SNRIn) % uncomment for parallel computing
% To reduce the total simulation time, you can execute this loop in
% parallel by using the Parallel Computing Toolbox. Comment out the 'for'
% statement and uncomment the 'parfor' statement. If the Parallel Computing
% Toolbox is not installed, 'parfor' defaults to normal 'for' statement

    % Set the random number generator settings to default values
    rng('default');

      
      
      %MaxDoppler=simParameters.DopplerIn(DopplerIdx);
      %simLocal.SNRIn(snrIdx);
    % Take full copies of the simulation-level parameter structures so that they are not 
    % PCT broadcast variables when using parfor 
    simLocal = simParameters; %参数
    waveinfoLocal = waveformInfo;%波形信息
    
    % Take copies of channel-level parameters to simply subsequent parameter referencing 
    carrier = simLocal.Carrier;
    pdsch = simLocal.PDSCH;
    pdschextra = simLocal.PDSCHExtension;
    ssburst = simLocal.SSBurst;
    decodeDLSCHLocal = decodeDLSCH;  % Copy of the decoder handle to help PCT classification of variable
    decodeDLSCHLocal.reset();        % Reset decoder at the start of each SNR point
    pathFilters = [];
    ssbWaveform = [];
     
    % Prepare simulation for new SNR point
    SNRdB = simLocal.SNRIn(snrIdx);
    fprintf('\nSimulating transmission scheme 1 (%dx%d) and SCS=%dkHz with %s channel at %gdB SNR for %d 10ms frame(s)\n',...
        simParameters.NTxAnts,simParameters.NRxAnts,carrier.SubcarrierSpacing, ...
        simLocal.DelayProfile,SNRdB,simLocal.NFrames); 
        
    % Initialize variables used in the simulation and analysis
    bitTput = [];           % Number of successfully received bits per transmission 收到的bit数
    txedTrBlkSizes = [];    % Number of transmitted info bits per transmission 每次发送的信息bit数
    
    % Specify the order in which we cycle through the HARQ processes
    harqSequence = 1:pdschextra.NHARQProcesses;

    % Initialize the state of all HARQ processes
    harqProcesses = hNewHARQProcesses(pdschextra.NHARQProcesses,rvSeq,pdsch.NumCodewords);
    harqProcCntr = 0; % HARQ process counter
        
    % Reset the channel so that each SNR point will experience the same
    % channel realization
    reset(channel);
    
    % Total number of slots in the simulation period
    NSlots = simLocal.NFrames * carrier.SlotsPerFrame;

    % Index to the start of the current set of SS burst samples to be
    % transmitted
    ssbSampleIndex = 1;
    
    % Obtain a precoding matrix (wtx) to be used in the transmission of the
    % first transport block
    estChannelGrid = getInitialChannelEstimate(carrier,simLocal.NTxAnts,channel); %信道估计    
    newWtx = getPrecodingMatrix(carrier,pdsch,estChannelGrid);%预编码
    
    % Timing offset, updated in every slot for perfect synchronization and
    % when the correlation is strong for practical synchronization
    offset = 0;

    % Loop over the entire waveform length 在整个波形 有800个slot 为啥？
    for nslot = 0:numslot-1
        
        % Update the carrier slot numbers for new slot
        carrier.NSlot = nslot;
        
        % Generate a new SS burst when necessary
        if (ssbSampleIndex==1)
            nSubframe = carrier.NSlot / carrier.SlotsPerSubframe;
            ssburst.NFrame = floor(nSubframe / 10);
            ssburst.NHalfFrame = mod(nSubframe / 5,2);
            [ssbWaveform,~,ssbInfo] = hSSBurst(ssburst);
        end
        
        % Get HARQ process index for the current PDSCH from HARQ index table
        harqProcIdx = harqSequence(mod(harqProcCntr,length(harqSequence))+1);
        
        % Update current HARQ process information (this updates the RV
        % depending on CRC pass or fail in the previous transmission for
        % this HARQ process)
        harqProcesses(harqProcIdx) = hUpdateHARQProcess(harqProcesses(harqProcIdx),pdsch.NumCodewords);
        
        % Calculate the transport block sizes for the codewords in the slot
        [pdschIndices,pdschIndicesInfo] = nrPDSCHIndices(carrier,pdsch);
        trBlkSizes = nrTBS(pdsch.Modulation,pdsch.NumLayers,numel(pdsch.PRBSet),pdschIndicesInfo.NREPerPRB,pdschextra.TargetCodeRate,pdschextra.XOverhead);
        
        % HARQ processing
        % Check CRC from previous transmission per codeword, i.e. is a retransmission required?
        for cwIdx = 1:pdsch.NumCodewords
            newdata = false;
            if harqProcesses(harqProcIdx).blkerr(cwIdx) % Error for last recorded decoding
                if (harqProcesses(harqProcIdx).RVIdx(cwIdx)==1) % Signals the start of the RV sequence
                    resetSoftBuffer(decodeDLSCHLocal,cwIdx-1,harqProcIdx-1);  % Explicit reset required in this case
                    newdata = true;
                end
            else    % No error
                newdata = true;
            end
            if newdata 
                trBlk = randi([0 1],trBlkSizes(cwIdx),1);
                setTransportBlock(encodeDLSCH,trBlk,cwIdx-1,harqProcIdx-1);
            end
        end
                                
        % Encode the DL-SCH transport blocks
        codedTrBlocks = encodeDLSCH(pdsch.Modulation,pdsch.NumLayers,...
            pdschIndicesInfo.G,harqProcesses(harqProcIdx).RV,harqProcIdx-1);
    
        % Get precoding matrix (wtx) calculated in previous slot
        wtx = newWtx;
        
        % Resource grid array
        pdschGrid = nrResourceGrid(carrier,simLocal.NTxAnts);
        
        % PDSCH modulation and precoding
        pdschSymbols = nrPDSCH(carrier,pdsch,codedTrBlocks);
        [pdschAntSymbols,pdschAntIndices] = hPRGPrecode(size(pdschGrid),carrier.NStartGrid,pdschSymbols,pdschIndices,wtx);
        
        % PDSCH mapping in grid associated with PDSCH transmission period
        pdschGrid(pdschAntIndices) = pdschAntSymbols;
        
        % PDSCH DM-RS precoding and mapping
        dmrsSymbols = nrPDSCHDMRS(carrier,pdsch);
        dmrsIndices = nrPDSCHDMRSIndices(carrier,pdsch);
        [dmrsAntSymbols,dmrsAntIndices] = hPRGPrecode(size(pdschGrid),carrier.NStartGrid,dmrsSymbols,dmrsIndices,wtx);        
        pdschGrid(dmrsAntIndices) = dmrsAntSymbols;

        % PDSCH PT-RS precoding and mapping
        ptrsSymbols = nrPDSCHPTRS(carrier,pdsch);
        ptrsIndices = nrPDSCHPTRSIndices(carrier,pdsch);
        [ptrsAntSymbols,ptrsAntIndices] = hPRGPrecode(size(pdschGrid),carrier.NStartGrid,ptrsSymbols,ptrsIndices,wtx);
        
        pdschGrid(ptrsAntIndices) = ptrsAntSymbols;        
        %pdschGrid是物理层发送的X 这里size是612*14是一个slot传输的数据
        % OFDM modulation of associated resource elements
        txWaveform = nrOFDMModulate(carrier, pdschGrid); %这里的信息就是导频和数据都有
        
        % Add the appropriate portion of SS burst waveform to the
        % transmitted waveform
        Nt = size(txWaveform,1);
        txWaveform = txWaveform + ssbWaveform(ssbSampleIndex + (0:Nt-1),:);
        ssbSampleIndex = mod(ssbSampleIndex + Nt,size(ssbWaveform,1));

        % Pass data through channel model. Append zeros at the end of the
        % transmitted waveform to flush channel content. These zeros take
        % into account any delay introduced in the channel. This is a mix
        % of multipath delay and implementation delay. This value may 
        % change depending on the sampling rate, delay profile and delay
        % spread
        txWaveform = [txWaveform; zeros(maxChDelay, size(txWaveform,2))];
        [rxWaveform,pathGains,sampleTimes] = channel(txWaveform);
        
        % Add AWGN to the received time domain waveform
        % Normalize noise power by the IFFT size used in OFDM modulation,
        % as the OFDM modulator applies this normalization to the
        % transmitted waveform. Also normalize by the number of receive
        % antennas, as the channel model applies this normalization to the
        % received waveform by default
        SNR = 10^(SNRdB/20); % Calculate linear noise gain
        N0 = 1/(sqrt(2.0*simLocal.NRxAnts*double(waveinfoLocal.Nfft))*SNR);
        noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
        rxWaveform = rxWaveform + noise;

        if (simLocal.PerfectChannelEstimator)
            % Perfect synchronization. Use information provided by the channel
            % to find the strongest multipath component
            pathFilters = getPathFilters(channel); % get path filters for perfect channel estimation
            [offset,mag] = nrPerfectTimingEstimate(pathGains,pathFilters);
        else
            % Practical synchronization. Correlate the received waveform 
            % with the PDSCH DM-RS to give timing offset estimate 't' and 
            % correlation magnitude 'mag'. The function hSkipWeakTimingOffset
            % is used to update the receiver timing offset. If the 
            % correlation peak in 'mag' is weak, the current timing
            % estimate 't' is ignored and the previous estimate 'offset'
            % is used
            [t,mag] = nrTimingEstimate(carrier,rxWaveform,dmrsIndices,dmrsSymbols); 
            offset = hSkipWeakTimingOffset(offset,t,mag);
            % Display a warning if the estimated timing offset exceeds the
            % maximum channel delay
            if offset > maxChDelay
                warning(['Estimated timing offset (%d) is greater than the maximum channel delay (%d).' ...
                    ' This will result in a decoding failure. This may be caused by low SNR,' ...
                    ' or not enough DM-RS symbols to synchronize successfully.'],offset,maxChDelay);
            end
        end
        rxWaveform = rxWaveform(1+offset:end, :);

        % Perform OFDM demodulation on the received data to recreate the
        % resource grid, including padding in the event that practical
        % synchronization results in an incomplete slot being demodulated
        rxGrid = nrOFDMDemodulate(carrier, rxWaveform);
        [K,L,R] = size(rxGrid);
        if (L < carrier.SymbolsPerSlot)
            rxGrid = cat(2,rxGrid,zeros(K,carrier.SymbolsPerSlot-L,R));
        end
        %这里是收 rxGrid 载波*14symbol*天线数目

            % Perfect channel estimation, using the value of the path gains
            % provided by the channel. This channel estimate does not
            % include the effect of transmitter precoding
            estChannelGridPer = nrPerfectChannelEstimate(carrier,pathGains,pathFilters,offset,sampleTimes);%信道估计的结果 612*14*4           
            estChannelPer = estChannelGridPer(1:600,3:14,:); %这里是完美信道估计的结果 前600*12*（） 
            size(estChannelPer)
            test = reshape(estChannelPer,24,25,12,4);
            test2 = permute(test,[2,3,1,4]);
            size(test)
            size(test2)
            size(Ideal_H(:,:,:,:,1))
            size(Ideal_H(:,:,:,:,2))
            size(Ideal_H(:,:,:,:,:))
            NSlots
            nslot
            return
            %H是 25*12*24*4 认为子载波是24个 符号是12个 所以每次能有25个
            Ideal_H(nslot*25+1:(nslot+1)*25,:,:,:,1) = real(permute(reshape(estChannelPer,24,25,12,4),[2,3,1,4]));%这里reshape成25*12*24*4
            Ideal_H(nslot*25+1:(nslot+1)*25,:,:,:,2) = imag(permute(reshape(estChannelPer,24,25,12,4),[2,3,1,4]));
            
            size(Ideal_H)
            return
            
            
            %disp(size(estChannelGrid));
            [estChannelGrid,noiseEst] = nrChannelEstimate(carrier,rxGrid,dmrsIndices,dmrsSymbols,'CDMLengths',pdsch.DMRS.CDMLengths); 
            estChannelLs = estChannelGrid(1:600,3:14,:); %这里是LS信道估计的结果 前600*12*（） 
            %H是 25*12*24*4 认为子载波是24个 符号是12个 所以每次能有25个
            Hls(nslot*25+1:(nslot+1)*25,:,:,:,1) = real(permute(reshape(estChannelLs,24,25,12,4),[2,3,1,4]));%这里reshape成25*12*24*4
            Hls(nslot*25+1:(nslot+1)*25,:,:,:,2) = imag(permute(reshape(estChannelLs,24,25,12,4),[2,3,1,4]));
            rxGridi=rxGrid(1:600,3:14,:);
            Received_Y(nslot*25+1:(nslot+1)*25,:,:,:,1)=real(permute(reshape(rxGridi,24,25,12,4),[2,3,1,4]));
            Received_Y(nslot*25+1:(nslot+1)*25,:,:,:,2)=imag(permute(reshape(rxGridi,24,25,12,4),[2,3,1,4]));   
            trGrid=pdschGrid(1:600,3:14);
            Ideal_X(nslot*25+1:(nslot+1)*25,:,:,1)=real(permute(reshape(trGrid,24,25,12),[2,3,1]));%12*24*25
            Ideal_X(nslot*25+1:(nslot+1)*25,:,:,2)=imag(permute(reshape(trGrid,24,25,12),[2,3,1]));
            X_detection=zeros(600,12);
            for j=1:1:12
                for i=1:1:600
                    h=squeeze(estChannelLs(i,j,:));
                    y=squeeze(rxGridi(i,j,:));
                    X_detection(i,j)=(h')*inv(h*h'+10^(-SNRdB/10)*eye(Nr))*y;
                end
            end
            X_detection_r=real(X_detection);
            X_detection_i=imag(X_detection);
            X_detection_r(X_detection_r>0)=sqrt(2)/2;
            X_detection_r(X_detection_r<0)=-sqrt(2)/2;
            X_detection_i(X_detection_i>0)=sqrt(2)/2;
            X_detection_i(X_detection_i<0)=-sqrt(2)/2;
            Transmit_X(nslot*25+1:(nslot+1)*25,:,:,1)=permute(reshape(X_detection_r,24,25,12),[2,3,1]);
            Transmit_X(nslot*25+1:(nslot+1)*25,:,:,2)=permute(reshape(X_detection_i,24,25,12),[2,3,1]);
            disp(nslot);

    end
   Ideal_X=(1-sign(Ideal_X))/2;
    save(['./Datasets/',simParameters.DelayProfile,num2str(SNRdB),'dB_',...
        num2str(simParameters.MaximumDopplerShift),'Hz_R_',num2str(simParameters.NRxAnts),'.mat'],...
        'Ideal_H','Hls','Received_Y','Ideal_X','Transmit_X');  
end

%% Local Functions

function validateNumLayers(simParameters)
% Validate the number of layers, relative to the antenna geometry

    nlayers = simParameters.PDSCH.NumLayers;
    ntxants = simParameters.NTxAnts;
    nrxants = simParameters.NRxAnts;
    antennaDescription = sprintf('min(NTxAnts,NRxAnts) = min(%d,%d) = %d',ntxants,nrxants,min(ntxants,nrxants));
    if nlayers > min(ntxants,nrxants)
        error('The number of layers (%d) must satisfy NLayers <= %s',...
            nlayers,antennaDescription);
    end
    
    % Display a warning if the maximum possible rank of the channel equals
    % the number of layers
    if (nlayers > 2) && (nlayers == min(ntxants,nrxants))
        warning(['The maximum possible rank of the channel, given by %s, is equal to NLayers (%d).' ...
            ' This may result in a decoding failure under some channel conditions.' ...
            ' Try decreasing the number of layers or increasing the channel rank' ...
            ' (use more transmit or receive antennas).'],antennaDescription,nlayers); %#ok<SPWRN>
    end

end

function estChannelGrid = getInitialChannelEstimate(carrier,nTxAnts,propchannel)
% Obtain channel estimate before first transmission. This can be used to
% obtain a precoding matrix for the first slot.

    ofdmInfo = nrOFDMInfo(carrier);
    
    chInfo = info(propchannel);
    maxChDelay = ceil(max(chInfo.PathDelays*propchannel.SampleRate)) + chInfo.ChannelFilterDelay;
    
    % Temporary waveform (only needed for the sizes)
    tmpWaveform = zeros((ofdmInfo.SampleRate/1000/carrier.SlotsPerSubframe)+maxChDelay,nTxAnts);
    
    % Filter through channel    
    [~,pathGains,sampleTimes] = propchannel(tmpWaveform);
    
    % Perfect timing synch    
    pathFilters = getPathFilters(propchannel);
    offset = nrPerfectTimingEstimate(pathGains,pathFilters);
    
    % Perfect channel estimate
    estChannelGrid = nrPerfectChannelEstimate(carrier,pathGains,pathFilters,offset,sampleTimes);
    
end

function wtx = getPrecodingMatrix(carrier,pdsch,hestGrid)
% Calculate precoding matrices for all PRGs in the carrier that overlap
% with the PDSCH allocation
    
    % Maximum CRB addressed by carrier grid
    maxCRB = carrier.NStartGrid + carrier.NSizeGrid - 1;
    
    % PRG size
    if (isfield(pdsch,'PRGBundleSize') && ~isempty(pdsch.PRGBundleSize))
        Pd_BWP = pdsch.PRGBundleSize;
    else
        Pd_BWP = maxCRB + 1;
    end
    
    % PRG numbers (1-based) for each RB in the carrier grid
    NPRG = ceil((maxCRB + 1) / Pd_BWP);
    prgset = repmat((1:NPRG),Pd_BWP,1);
    prgset = prgset(carrier.NStartGrid + (1:carrier.NSizeGrid).');
    
    [~,~,R,P] = size(hestGrid);
    wtx = zeros([pdsch.NumLayers P NPRG]);
    for i = 1:NPRG
    
        % Subcarrier indices within current PRG and within the PDSCH
        % allocation
        thisPRG = find(prgset==i) - 1;
        thisPRG = intersect(thisPRG,pdsch.PRBSet(:) + carrier.NStartGrid,'rows');
        prgSc = (1:12)' + 12*thisPRG';
        prgSc = prgSc(:);
       
        if (~isempty(prgSc))
        
            % Average channel estimate in PRG
            estAllocGrid = hestGrid(prgSc,:,:,:);
            Hest = permute(mean(reshape(estAllocGrid,[],R,P)),[2 3 1]);

            % SVD decomposition
            [~,~,V] = svd(Hest);
            wtx(:,:,i) = V(:,1:pdsch.NumLayers).';

        end
    
    end
    
    wtx = wtx / sqrt(pdsch.NumLayers); % Normalize by NumLayers

end

function estChannelGrid = precodeChannelEstimate(carrier,estChannelGrid,W)
% Apply precoding matrix W to the last dimension of the channel estimate

    [K,L,R,P] = size(estChannelGrid);
    estChannelGrid = reshape(estChannelGrid,[K*L R P]);
    estChannelGrid = hPRGPrecode([K L R P],carrier.NStartGrid,estChannelGrid,reshape(1:numel(estChannelGrid),[K*L R P]),W);
    estChannelGrid = reshape(estChannelGrid,K,L,R,[]);

end

function [mappedPRB,mappedSymbols] = mapNumerology(subcarriers,symbols,nrbs,nrbt,fs,ft)
% Map the SSBurst numerology to PDSCH numerology. The outputs are:
%   - mappedPRB: 0-based PRB indices for carrier resource grid (arranged in a column)
%   - mappedSymbols: 0-based OFDM symbol indices in a slot for carrier resource grid (arranged in a row)
% The input parameters are:
%   - subcarriers: 1-based row subscripts for SSB resource grid (arranged in a column)
%   - symbols: 1-based column subscripts for SSB resource grid (arranged in an N-by-4 matrix, 4 symbols for each transmitted burst in a row, N transmitted bursts)
%     SSB resource grid is sized using ssbInfo.NRB, normal CP, spanning 5 subframes
%   - nrbs: source (SSB) NRB
%   - nrbt: target (carrier) NRB
%   - fs: source (SSB) SCS
%   - ft: target (carrier) SCS

    mappedPRB = unique(fix((subcarriers-(nrbs*6) - 1)*fs/(ft*12) + nrbt/2),'stable');
    
    symbols = symbols.';
    symbols = symbols(:).' - 1;

    if (ft < fs)
        % If ft/fs < 1, reduction
        mappedSymbols = unique(fix(symbols*ft/fs),'stable');
    else
        % Else, repetition by ft/fs
        mappedSymbols = reshape((0:(ft/fs-1))' + symbols(:)'*ft/fs,1,[]);
    end
    
end

function plotLayerEVM(NSlots,nslot,pdsch,siz,pdschIndices,pdschSymbols,pdschEq)
% Plot EVM information

    persistent slotEVM;
    persistent rbEVM
    persistent evmPerSlot;
    
    if (nslot==0)
        slotEVM = comm.EVM;
        rbEVM = comm.EVM;
        evmPerSlot = NaN(NSlots,pdsch.NumLayers);
        figure;
    end
    evmPerSlot(nslot+1,:) = slotEVM(pdschSymbols,pdschEq);
    subplot(2,1,1);
    plot(0:(NSlots-1),evmPerSlot);
    xlabel('Slot number');
    ylabel('EVM (%)');
    legend("layer " + (1:pdsch.NumLayers),'Location','EastOutside');
    title('EVM per layer per slot');

    subplot(2,1,2);
    [k,~,p] = ind2sub(siz,pdschIndices);
    rbsubs = floor((k-1) / 12);
    NRB = siz(1) / 12;
    evmPerRB = NaN(NRB,pdsch.NumLayers);
    for nu = 1:pdsch.NumLayers
        for rb = unique(rbsubs).'
            this = (rbsubs==rb & p==nu);
            evmPerRB(rb+1,nu) = rbEVM(pdschSymbols(this),pdschEq(this));
        end
    end
    plot(0:(NRB-1),evmPerRB);
    xlabel('Resource block');
    ylabel('EVM (%)');
    legend("layer " + (1:pdsch.NumLayers),'Location','EastOutside');
    title(['EVM per layer per resource block, slot #' num2str(nslot)]);
    
    drawnow;
    
end