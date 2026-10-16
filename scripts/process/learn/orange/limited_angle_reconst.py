method = 'iterative'
deltas = [5, 10, 20, 45, 90, 180]
deltas = [5]
# deltas = [5]

do_simulate = True
do_reconstruct = True
do_cuda = True  # use CUDA image/forward-projector if available

for delta in deltas:
    print(delta)
    """
    s1_tomosynthesis.py
    Author: Owen Dillon, The University of Sydney (Python/RTK port)
    Purpose: Tomosynthesis simulation + ADMM-TV reconstruction using RTK's
             Python bindings (itk.RTK), replacing MATLAB + system() calls.
    
    Requires: itk, itk-rtk (pip install itk itk-rtk-embedded / conda install rtk)
    """
    
    import os
    import shutil
    import numpy as np
    import itk
    from itk import RTK as rtk
    import time
    
    # # ---------------------------------------------------------------
    # # File locations
    # # ---------------------------------------------------------------
    # name_gt = r"D:\Brett\code\Tomosynthesis\xcat_chest.mha"
    name_gt = 'files:learn/orange/lung/pat01/cbct.mha'
    name_gt = resolve_filepath(name_gt)
    # name_gt = r"D:\Brett\code\Tomosynthesis\xcat_chest.mha"
    
    # ---------------------------------------------------------------
    # Reconstruction volume
    # ---------------------------------------------------------------
    recon_size = [540, 270, 350]
    recon_spacing = [1.0, 1.0, 1.0]
    recon_origin = [
        -(recon_size[0] * recon_spacing[0]) / 2,
         -(recon_size[1] * recon_spacing[1]) / 2,
         -(recon_size[2] * recon_spacing[2]) / 2
    ]
    
    # ---------------------------------------------------------------
    # Geometry: theta = mid-arc angle, delta = half-arc extent (deg)
    # n_proj projections spanning [theta-delta, theta+delta]
    # ---------------------------------------------------------------
    SDD = info.iloc[0]['sdd']
    SID = info.iloc[0]['sid']
    theta = 0.0
    # n_proj = int(2 * delta + 1)
    n_proj = infos[delta]['n_proj']
    # angs = np.linspace(theta - delta, theta + delta, n_proj)
    angs = infos[delta]['proj_angles']
    dir_out = fr"D:\Brett\code\Tomosynthesis\orange_outputs_{int(delta)}"
    
    # Detector
    n1, n2 = info.iloc[0]['det-size']
    det_spacing = info.iloc[0]['det-spacing']
    det_offset = info.iloc[0]['det-offset']
    
    os.makedirs(dir_out, exist_ok=True)
    
    CpuImageType = itk.Image[itk.F, 3]
    # RTK's CUDA filters (e.g. rtkCudaForwardProjectionImageFilterCIF3) are
    # templated on itk.CudaImage, not itk.Image. Use CudaImage end-to-end for
    # the GPU path; only cast back to a plain Image where a CPU-only step
    # (numpy conversion, file writing) needs it.
    ImageType = itk.CudaImage[itk.F, 3] if do_cuda else CpuImageType
    
    # ---------------------------------------------------------------
    # Build geometry
    # ---------------------------------------------------------------
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for ang in angs:
        # AddProjection(SID, SDD, GantryAngle, ...) — out-of-plane/in-plane
        # offsets left at 0 to mirror the MATLAB G vector [ang,0,0,0,0,0,0,SID,SDD]
        geometry.AddProjection(SID, SDD, float(ang), det_offset[0], det_offset[1], 0, 0, 0, 0)
    
    geom_writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
    geom_writer.SetFilename(os.path.join(dir_out, "geo.xml"))
    geom_writer.SetObject(geometry)
    print("Writing geometry file")
    geom_writer.WriteFile()
    print("Written!")
    
    # ---------------------------------------------------------------
    # Simulate projections (forward projection through ground-truth volume)
    # ---------------------------------------------------------------
    if do_simulate:
        start = time.perf_counter()
        print('Reading GT')
        gt_cpu = itk.imread(name_gt, itk.F)
        print('Read.')
        end = time.perf_counter()
        print(f"Elapsed: {end - start:.3f} s")
    
        if do_cuda:
            # CastImageFilter isn't wrapped for CudaImage output, and Graft()
            # requires an argument of the exact same image type, so graft a
            # CudaImage view directly onto the CPU image's buffer field-by-field.
            gt = ImageType.New()
            gt.SetPixelContainer(gt_cpu.GetPixelContainer())
            gt.CopyInformation(gt_cpu)
            gt.SetBufferedRegion(gt_cpu.GetBufferedRegion())
            gt.SetRequestedRegion(gt_cpu.GetRequestedRegion())
            # This image has no generating filter, so it can never regenerate
            # its buffer once released — keep RTK from freeing it between passes.
            gt.ReleaseDataFlagOff()
        else:
            gt = gt_cpu
    
        constant_image_source = rtk.ConstantImageSource[ImageType].New()
        constant_image_source.SetOrigin(
            [-(n1 / 2) * det_spacing[0], -(n2 / 2) * det_spacing[1], 1.0]
        )
        constant_image_source.SetSpacing([det_spacing[0], det_spacing[1], 1.0])
        constant_image_source.SetSize([n1, n2, n_proj])
        constant_image_source.SetConstant(0.0)
    
        if do_cuda:
            fwd_proj = rtk.CudaForwardProjectionImageFilter[ImageType].New()
        else:
            fwd_proj = rtk.JosephForwardProjectionImageFilter[ImageType, ImageType].New()
    
        fwd_proj.SetInput(0, constant_image_source.GetOutput())
        fwd_proj.SetInput(1, gt)
        fwd_proj.SetGeometry(geometry)
        print("Forward projecting!")
        fwd_proj.Update()
        drr = fwd_proj.GetOutput()
        print("All done.")
    
        if do_cuda:
            drr.GetCudaDataManager().UpdateCPUBuffer()
            # itk.PyBuffer (used by array_from_image) isn't wrapped for
            # CudaImage. Graft a plain CPU Image view onto the same buffer
            # (mirror of the CPU->GPU graft used for gt/proj_gpu above).
            drr_cpu = CpuImageType.New()
            drr_cpu.SetPixelContainer(drr.GetPixelContainer())
            drr_cpu.CopyInformation(drr)
            drr_cpu.SetBufferedRegion(drr.GetBufferedRegion())
            drr_cpu.SetRequestedRegion(drr.GetRequestedRegion())
        else:
            drr_cpu = drr
    
        proj_dir = os.path.join(dir_out, "proj")
        os.makedirs(proj_dir, exist_ok=True)
    
        # Write raw float32 line integrals directly — unlike HNC, RTK's mha reader
        # applies no log transform, so ADMM must receive line integrals not transmission.
        drr_arr = itk.array_from_image(drr_cpu)
        for kk in range(n_proj):
            proj_img = itk.image_from_array(drr_arr[kk, :, :].astype(np.float32))
            proj_img.SetSpacing([det_spacing[0], det_spacing[1]])
            proj_img.SetOrigin([-(n1 / 2) * det_spacing[0], -(n2 / 2) * det_spacing[1]])
            itk.imwrite(proj_img, os.path.join(proj_dir, f"proj_{kk+1:05d}.mha"))
    
    if do_reconstruct:
        # ProjectionsReader is likely not wrapped for CudaImage either — read
        # via the CPU type, then graft onto a GPU view (same idiom as GT above).
        proj_stack = rtk.ProjectionsReader[CpuImageType].New()
        proj_files = sorted(
            os.path.join(dir_out, "proj", f) for f in os.listdir(os.path.join(dir_out, "proj"))
        )
        proj_stack.SetFileNames(proj_files)
        proj_stack.Update()
        proj_cpu = proj_stack.GetOutput()
        
        if do_cuda:
            proj_gpu = ImageType.New()
            proj_gpu.SetPixelContainer(proj_cpu.GetPixelContainer())
            proj_gpu.CopyInformation(proj_cpu)
            proj_gpu.SetBufferedRegion(proj_cpu.GetBufferedRegion())
            proj_gpu.SetRequestedRegion(proj_cpu.GetRequestedRegion())
            # ADMM's AL iterations re-request these projections repeatedly;
            # since this orphan image has no source to regenerate from, it must
            # never be released or its buffer collapses to size 0 on reuse.
            proj_gpu.ReleaseDataFlagOff()
        else:
            proj_gpu = proj_cpu
        
        # Initial-estimate volume for ADMM (input 0) — MUST be sized to the
        # reconstruction geometry (recon_size/recon_spacing), not left default,
        # or the filter crashes natively during Update().
        recon_source = rtk.ConstantImageSource[ImageType].New()
        recon_source.SetSize(recon_size)
        recon_source.SetSpacing(recon_spacing)
        recon_source.SetOrigin(recon_origin)
        recon_source.SetConstant(0.0)
        
        if method == 'iterative':
            admm = rtk.ADMMTotalVariationConeBeamReconstructionFilter[ImageType].New()
            admm.SetInput(0, recon_source.GetOutput())
            admm.SetInput(1, proj_gpu)
            admm.SetGeometry(geometry)
        
            # $\alpha$: TV regularisation weight, $\beta$: ADMM penalty parameter
            admm.SetAlpha(0.03)
            admm.SetBeta(10.0)
            admm.SetAL_iterations(10)
            admm.SetCG_iterations(5)
        
            if do_cuda:
                admm.SetForwardProjectionFilter(admm.ForwardProjectionType_FP_CUDARAYCAST)
                admm.SetBackProjectionFilter(admm.BackProjectionType_BP_CUDAVOXELBASED)
            admm.SetDisableDisplacedDetectorFilter(True)
        
            import time
        
            start = time.perf_counter()
            print("Running iterative method")
            admm.Update()
            print("Completed")
            end = time.perf_counter()
            print(f"Elapsed: {end - start:.3f} s")
        
            # CudaImage is-a itk.Image (by inheritance), so its output can be
            # passed directly to a CPU-templated writer — RTK syncs the GPU
            # buffer back to CPU automatically. No cast filter needed.
            writer = itk.ImageFileWriter[CpuImageType].New()
            writer.SetFileName(os.path.join(dir_out, "iterative_python.mha"))
            writer.SetInput(admm.GetOutput())
            writer.Update()
        
        elif method == 'fdk':
        
            # FDK filter parameters
            hann_cut_frequency = 0.2    # Hann window cutoff (0 = ramp only, 1 = full Hann)
            truncation_correction = 0.0  # lateral truncation padding (0 to 1)
            
            if do_cuda:
                fdk = rtk.CudaFDKConeBeamReconstructionFilter[ImageType].New()
            else:
                fdk = rtk.FDKConeBeamReconstructionFilter[ImageType].New()
            
            fdk.SetInput(0, recon_source.GetOutput())
            fdk.SetInput(1, proj_gpu)
            fdk.SetGeometry(geometry)
            fdk.GetRampFilter().SetTruncationCorrection(truncation_correction)
            fdk.GetRampFilter().SetHannCutFrequency(hann_cut_frequency)
            
            start = time.perf_counter()
            print("Running FDK")
            fdk.Update()
            print(f"Completed. Elapsed: {time.perf_counter() - start:.3f} s")
            
            writer = itk.ImageFileWriter[CpuImageType].New()
            writer.SetFileName(os.path.join(dir_out, "fdk_python.mha"))
            writer.SetInput(fdk.GetOutput())
            writer.Update()
    
    # # ---------------------------------------------------------------
    # # Cleanup temporary projection directory
    # # ---------------------------------------------------------------
    # proj_dir = os.path.join(dir_out, "proj")
    # if os.path.isdir(proj_dir):
    #     shutil.rmtree(proj_dir)
