#!/usr/local/bin/python3.6
# coding: utf-8

import cgitb
cgitb.enable()

from wsgiref.handlers import CGIHandler
from app import app
CGIHandler().run(app)