from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info

class MyTopo( Topo ):
    "Simple topology example."

    def build( self ):
        "Create custom topo."

        # Add hosts and switches
        host1 = self.addHost( 'h1', ip='192.168.2.10' )
        host2 = self.addHost( 'h2', ip='192.168.2.20' )
        host3 = self.addHost( 'h3', ip='192.168.2.30' )
        host4 = self.addHost( 'h4', ip='192.168.2.40' )
        switch1 = self.addSwitch( 's1' )

        # Add links
        self.addLink(switch1, host1 )
        self.addLink(switch1, host2 )
        self.addLink(switch1, host3 )
        self.addLink(switch1, host4 )



topos = { 'mytopo': ( lambda: MyTopo() ) }

#commnad to run it with 
# sudo mn --custom testTopo.py --topo mytopo --controller=remote,port=6633 #--controller=remote,port=6655 --switch=ovsk --mac
# once up, run these two commnads
#py net.addLink('c1','s1')
#py net.addLink('c0','s1') 


