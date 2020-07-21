#!/usr/bin/python

# Plotting packages for neutpy

# TODO: Actually implement this unless they will just be neutpy class methods

#
# fig_width = 6.0
# fig_height = (np.amax(inst.Z) - np.amin(inst.Z)) / (np.amax(inst.R) - np.amin(inst.R)) * fig_width
#
# fig1 = plt.figure(figsize=(0.975*fig_width, fig_height))
# ax1 = fig1.add_subplot(1, 1, 1)
# ax1.axis('equal')

# psi raw
# ax1.imshow(np.flipud(inst.psi), aspect='auto')

# ax1.contourf(inst.R, inst.Z, inst.psi, 500)

# Br = np.gradient(inst.psi, axis=1)/inst.R
# Bz = -np.gradient(inst.psi, axis=0)/inst.R
# B_p = np.sqrt((np.gradient(inst.psi, axis=1)/inst.R)**2 + \
#                 (-np.gradient(inst.psi, axis=0)/inst.R)**2)
# ax1.contourf(inst.R, inst.Z, B_p, 500)

#ax1.plot(inst.wallx, inst.wally, color='black', lw=1, zorder=10)

# for i in range(int(len(inst.dpsidR_0)/2)):
#     x1, y1 = np.split(inst.dpsidR_0[i], 2, axis=1)
#     # ax1.plot(x1, y1, color='red', lw=1, label = 'dpsidR=0')
# for i in range(int(len(inst.dpsidZ_0)/2)):
#     x1, y1 = np.split(inst.dpsidZ_0[i], 2, axis=1)
    # ax1.plot(x1, y1, color='blue', lw=1, label = 'dpsidZ=0')
# ax1.legend()

# xpt and mag_axis
# ax1.scatter(inst.xpt[0], inst.xpt[1], color='yellow', zorder=10)
# ax1.scatter(inst.m_axis[0], inst.m_axis[1], color='yellow', zorder=10)

# core density
# R, Z = np.meshgrid(np.linspace(1.02, 2.31, 500), np.linspace(-1.15, 0.95, 500))
# ni = griddata(inst.ni_pts[:, :-1], inst.ni_pts[:, 2], (R, Z), method='cubic', fill_value=inst.ni_pts[-1, 2]*1.0)
# dnidr = np.abs(np.gradient(ni, axis=1)) + np.abs(np.gradient(ni, axis=0))
# ax1.contourf(R, Z, dnidr, 500)

#
# # seperatrix
# coords = np.asarray(inst.main_sep_line.coords)
# coords = np.vstack((coords, coords[0]))
# ax1.plot(coords[:, 0], coords[:, 1], color='yellow', lw=1)
#
# # inboard divertor
# coords = np.asarray(inst.ib_div_line_cut.coords)
# ax1.plot(coords[:, 0], coords[:, 1], color='yellow', lw=1)
#
# # outboard divertor
# coords = np.asarray(inst.ob_div_line_cut.coords)
# ax1.plot(coords[:, 0], coords[:, 1], color='yellow', lw=1)
#
# # plot core lines
# for i, line in enumerate(inst.core_lines):
#     coords = np.asarray(line.coords)
#     coords = np.vstack((coords, coords[0]))
#     # ax1.plot(coords[:, 0], coords[:, 1], color='pink', lw=1)
#
# # plot sol lines
# for i, line in enumerate(inst.sol_lines_cut):
#     coords = np.asarray(line.coords)
#     # ax1.plot(coords[:, 0], coords[:, 1], color='lime', lw=1)
