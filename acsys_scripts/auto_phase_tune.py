import numpy as np
import acsys.dpm


async def my_app(con):
    # Setup context
    async with acsys.dpm.DPMContext(con, dpm_node='DPM02') as dpm:
        # Check kerberos credentials and enable settings
        await dpm.enable_settings(role='linac_daily_rf_tuning')

        # Add acquisition requests
        #await dpm.add_entry(0, 'L:V5QSET.SETTING@e,52,e,0')
        await dpm.add_entry(0, 'L:RFBPAH.SETTING@e,52,e,0')
        await dpm.add_entry(1, 'L:D7LMSM@e,52,e,0')

        # Define range
        maxval = 228.5
        minval = 227.5

        # Init
        i=0
        step=0.1
        minloss=100
        minidx=-1
        avgvec =[]

        await dpm.apply_settings([(0, maxval)])
        await dpm.start()
        async for ii in dpm.replies():
            if ii.isReadingFor(0):
                if(ii.data <= minval):
                    break
                if(i%10==0):
                    setval = ii.data-step
                    print(i, 'read 0:',ii.data, 'set 0:',setval)
                    await dpm.apply_settings([(0, setval)])


            elif ii.isReadingFor(1):
                avgvec.append(ii.data)
                if(i%10==0):
                    avg = np.mean(avgvec)
                    if(avg<minloss):
                        minloss = avg
                        minidx = i
                    avgvec=[]
                    print(i, 'read 1 (avg):', avg, 'minloss:',minloss, 'minidx',minidx)
                i=i+1
                
            else:
                pass

        await dpm.apply_settings([(0, maxval-minidx/10*step)])
        
        
acsys.run_client(my_app)
