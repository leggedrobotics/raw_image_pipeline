# Alphasense software setup (host PC)

The instructions below must be executed in the host PC (i.e, the PC where you will connect the Alphasense).


## Install Sevensense's Alphasense drivers

The drivers you install in the host PC must match the firmware version on Alphasense you're using. If not, you'll get:

1. Host errors when trying to launch the drivers (usually because the driver is older than the firmware)
2. Firmware errors asking to update it (when the firmware is older than the drivers)

Bear this in mind before installing the drivers on your host PC.

### Latest version of drivers:

```sh
curl -Ls https://deb.7sr.ch/pubkey.gpg | sudo apt-key add -
echo "deb [arch=amd64] https://deb.7sr.ch/alphasense focal main" | sudo tee /etc/apt/sources.list.d/sevensense.list
sudo apt update
sudo apt install alphasense-driver-core alphasense-firmware ros-noetic-alphasense-driver-ros ros-noetic-alphasense-driver
echo -e "net.core.rmem_max=11145728" | sudo tee /etc/sysctl.d/90-increase-network-buffers.conf
```

### SubT/Glimpse-compatible drivers:
We need to change the PPA to `alphasense/1.0` and manually install the compatible version:

```sh
curl -Ls https://deb.7sr.ch/pubkey.gpg | sudo apt-key add -
echo "deb [arch=amd64] https://deb.7sr.ch/alphasense/1.0 focal main" | sudo tee /etc/apt/sources.list.d/sevensense.list
sudo apt update
sudo apt install alphasense-driver-core=1.7.2-0~build83~focal \
                 alphasense-firmware=1.7.2-0~build82~focal \
                 ros-noetic-alphasense-driver-ros=1.7.2-0~build81~focal \
                 ros-noetic-alphasense-driver=1.7.2-0~build79~focal
echo -e "net.core.rmem_max=11145728" | sudo tee /etc/sysctl.d/90-increase-network-buffers.conf
```

### Alphasense viewer

If you also need the alphasense viewer:

### Latest
```sh
sudo apt install alphasense-viewer
```

### SubT/Glimpse
```sh
sudo apt install alphasense-viewer=1.7.2-0~build81~focal
```

## Network configuration

### Identify the ethernet interface
Identify the ethernet interface (we will call it `INTERFACE_NAME_ALPHASENSE` onwards) you'll use to connect to the Alphasense. e.g `enp4s0`, `eno0`, `eth0`

### PTP and PHC2SYS Configurations

Instructions on how to setup PTP time sync; steps to be executed on NPC:

```
sudo apt install linuxptp ethtool
```

Modify `/etc/linuxptp/ptp4l.conf` and
(A) Change line regarding `clockClass` to:
```
clockClass		128
```

(B) append the following lines at the end of the file, replacing `INTERFACE_NAME_ALPHASENSE` by the interface name:
```sh
# RSL Customization
boundary_clock_jbod 1
[INTERFACE_NAME_ALPHASENSE]
```

Example:
```sh
# RSL Customization
boundary_clock_jbod 1
[enp4s0]
```

Create a systemd drop-in directory to override the system service file:
```
sudo mkdir -p /etc/systemd/system/ptp4l.service.d
```

Create a file at `/etc/systemd/system/ptp4l.service.d/override.conf` with the following contents:
```sh
[Service]
ExecStart=
ExecStart=/usr/sbin/ptp4l -f /etc/linuxptp/ptp4l.conf
```

Restart the ptp4l service so the change takes effect:
```sh
sudo systemctl daemon-reload
sudo systemctl restart ptp4l
sudo systemctl status ptp4l
```

Enable the ptp4l service:
```sh
sudo systemctl enable ptp4l.service
```

## Alphasense - phc2sys
Create a systemd drop-in directory to override the system service file:
```sh
sudo mkdir -p /etc/systemd/system/phc2sys.service.d
```

Create a file at `/etc/systemd/system/phc2sys.service.d/override.conf` with the following contents, replacing `INTERFACE_NAME_ALPHASENSE` by the interface name:
```sh
[Service]
ExecStart=
ExecStart=/usr/sbin/phc2sys -c INTERFACE_NAME_ALPHASENSE -s CLOCK_REALTIME -w -O 0
```

Restart the phc2sys service so the change takes effect:
```sh
sudo systemctl daemon-reload
sudo systemctl restart phc2sys
sudo systemctl status phc2sys
```

Enable the phc2sys service:
```sh
sudo systemctl enable phc2sys.service
```

## Recommendations
### Maximize Network Performance
Follow instructions at: [maximize_network_performance.md](https://github.com/sevensense-robotics/alphasense_core_manual/blob/master/pages/maximize_network_performance.md)

Hints:
To permanently set 'mtu' to 7260 (see `mtu: XXX`) of ethernet interface where alphasense is connected, netplan config is modified as follow:

```sh
network:
    version: 2
    renderer: networkd
    ethernets:
        enp4s0:
            dhcp4: no
            mtu: XXX # this is where you can set mtu in netplan config file
            addresses: [192.168.20.1/24]
```

### Increase Network Buffer
```sh
echo -e "net.core.rmem_max=11145728" | sudo tee /etc/sysctl.d/90-increase-network-buffers.conf
```
Above command taken from "getting started" github [page](https://github.com/sevensense-robotics/alphasense_core_manual/blob/master/pages/getting_started.md#1-launching-the-standalone-viewer).

Settings for ANYmal Cerberus (23/11/2020)
```sh
- net.core.netdev_budget=600
- net.core.netdev_max_backlog=2000
- mtu 7260
- net.core.rmem_max=11145728
```

### Time Synchronization
The settings for [time_synchronization.md](https://github.com/sevensense-robotics/alphasense_core_manual/blob/master/pages/time_synchronization.md) over PTP are included in the section [PTP and PHC2SYS Configurations](###PTP-and-PHC2SYS-Configurations).
