# -*- coding: utf-8 -*-
"""
/***************************************************************************
 PointProcessSimulator
                                 A QGIS plugin
 Estimates the intensity of a point process (via kernel density estimation) and resimulates an inhomogeneous Poisson-process according to this intensity.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2020-02-27
        copyright            : (C) 2020 by Mathias Weiße
        email                : weisse.m@posteo.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""

__author__ = 'Mathias Weiße'
__date__ = '2020-02-27'
__copyright__ = '(C) 2020 by Mathias Weiße'


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load PointProcessSimulator class from file PointProcessSimulator.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .point_process_simulator import PointProcessSimulatorPlugin
    return PointProcessSimulatorPlugin()
