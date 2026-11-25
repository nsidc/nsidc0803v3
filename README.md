<p align="center">
  <img alt="NSIDC logo" src="https://nsidc.org/themes/custom/nsidc/logo.svg" width="150" />
</p>


# NSIDC0803v3

{title} enables {audience} to {utility}.


## Level of Support

* This repository is fully supported by NSIDC. If you discover any problems or bugs,
  please submit an Issue. If you would like to contribute to this repository, you may fork
  the repository and submit a pull request. 
* This repository is not actively supported by NSIDC but we welcome issue submissions and
  pull requests in order to foster community contribution.

See the [LICENSE](LICENSE) for details on permissions and warranties. Please contact
nsidc@nsidc.org for more information.


## Requirements

Conda


## Installation

{installation}


## Usage

These routines expect files created by nsidc0802v3 to be available to it. 
Currently, those files are on `/share/apps/nsidc0802/nsidc0802/dev/scotts/<...>/

From the main repo directory here, the initial means of running the code is:

`python seaice_goddard.py <gridid> <startdate> <enddate>`

eg

`python seaice_goddard.py psn25 20240315 20240315`

creates:  `/share/apps/nsidc0803/nsidc0803/dev/scotts/nt_0803v3/2024/03/nt_0803v3_psn25_20240315.dat`


`python seaice_goddard.py pss25 20240315 20240315`

creates: `/share/apps/nsidc0803/nsidc0803/dev/scotts/nt_0803v3/2024/03/nt_0803v3_pss25_20240315.dat`


## Troubleshooting

{troubleshooting}


## Credit

This content was developed by the National Snow and Ice Data Center with funding from
multiple sources.
