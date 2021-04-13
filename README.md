# qc-backtest-coordinator

See environment.yml for package requirements (can be used to configure a Conda env)

## Instructions

- Create a working directory and copy config.json there.
- Update config.json fields to correct values.
- Edit string in main.py to point to your working dir
- Create a python module which exposes an attribute that is of type TestSet or any of its subclasses
- Update config.json with your module and attribute name
- Run main.py
