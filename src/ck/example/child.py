from ck.pgm import PGM


class Child(PGM):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        pgm_rv0 = self.new_rv('BirthAsphyxia', ('yes', 'no'))
        pgm_rv1 = self.new_rv('HypDistrib', ('Equal', 'Unequal'))
        pgm_rv2 = self.new_rv('HypoxiaInO2', ('Mild', 'Moderate', 'Severe'))
        pgm_rv3 = self.new_rv('CO2', ('Normal', 'Low', 'High'))
        pgm_rv4 = self.new_rv('ChestXray', ('Normal', 'Oligaemic', 'Plethoric', 'Grd_Glass', 'Asy/Patch'))
        pgm_rv5 = self.new_rv('Grunting', ('yes', 'no'))
        pgm_rv6 = self.new_rv('LVHreport', ('yes', 'no'))
        pgm_rv7 = self.new_rv('LowerBodyO2', ('<5', '5-12', '12+'))
        pgm_rv8 = self.new_rv('RUQO2', ('<5', '5-12', '12+'))
        pgm_rv9 = self.new_rv('CO2Report', ('<7.5', '>=7.5'))
        pgm_rv10 = self.new_rv('XrayReport', ('Normal', 'Oligaemic', 'Plethoric', 'Grd_Glass', 'Asy/Patchy'))
        pgm_rv11 = self.new_rv('Disease', ('PFC', 'TGA', 'Fallot', 'PAIVS', 'TAPVD', 'Lung'))
        pgm_rv12 = self.new_rv('GruntingReport', ('yes', 'no'))
        pgm_rv13 = self.new_rv('Age', ('0-3_days', '4-10_days', '11-30_days'))
        pgm_rv14 = self.new_rv('LVH', ('yes', 'no'))
        pgm_rv15 = self.new_rv('DuctFlow', ('Lt_to_Rt', 'None', 'Rt_to_Lt'))
        pgm_rv16 = self.new_rv('CardiacMixing', ('None', 'Mild', 'Complete', 'Transp.'))
        pgm_rv17 = self.new_rv('LungParench', ('Normal', 'Congested', 'Abnormal'))
        pgm_rv18 = self.new_rv('LungFlow', ('Normal', 'Low', 'High'))
        pgm_rv19 = self.new_rv('Sick', ('yes', 'no'))
        pgm_factor0 = self.new_factor(pgm_rv0)
        pgm_factor1 = self.new_factor(pgm_rv1, pgm_rv15, pgm_rv16)
        pgm_factor2 = self.new_factor(pgm_rv2, pgm_rv16, pgm_rv17)
        pgm_factor3 = self.new_factor(pgm_rv3, pgm_rv17)
        pgm_factor4 = self.new_factor(pgm_rv4, pgm_rv17, pgm_rv18)
        pgm_factor5 = self.new_factor(pgm_rv5, pgm_rv17, pgm_rv19)
        pgm_factor6 = self.new_factor(pgm_rv6, pgm_rv14)
        pgm_factor7 = self.new_factor(pgm_rv7, pgm_rv1, pgm_rv2)
        pgm_factor8 = self.new_factor(pgm_rv8, pgm_rv2)
        pgm_factor9 = self.new_factor(pgm_rv9, pgm_rv3)
        pgm_factor10 = self.new_factor(pgm_rv10, pgm_rv4)
        pgm_factor11 = self.new_factor(pgm_rv11, pgm_rv0)
        pgm_factor12 = self.new_factor(pgm_rv12, pgm_rv5)
        pgm_factor13 = self.new_factor(pgm_rv13, pgm_rv11, pgm_rv19)
        pgm_factor14 = self.new_factor(pgm_rv14, pgm_rv11)
        pgm_factor15 = self.new_factor(pgm_rv15, pgm_rv11)
        pgm_factor16 = self.new_factor(pgm_rv16, pgm_rv11)
        pgm_factor17 = self.new_factor(pgm_rv17, pgm_rv11)
        pgm_factor18 = self.new_factor(pgm_rv18, pgm_rv11)
        pgm_factor19 = self.new_factor(pgm_rv19, pgm_rv11)
        
        pgm_function0 = pgm_factor0.set_dense()
        pgm_function0.set_flat(0.1, 0.9)
        
        pgm_function1 = pgm_factor1.set_dense()
        pgm_function1.set_flat(
            0.95, 0.95, 0.95, 0.95, 0.95, 
            0.95, 0.95, 0.95, 0.05, 0.5, 
            0.95, 0.5, 0.05, 0.05, 0.05, 
            0.05, 0.05, 0.05, 0.05, 0.05, 
            0.95, 0.5, 0.05, 0.5
        )
        
        pgm_function2 = pgm_factor2.set_dense()
        pgm_function2.set_flat(
            0.93, 0.15, 0.7, 0.1, 0.1, 
            0.1, 0.1, 0.05, 0.1, 0.02, 
            0.1, 0.02, 0.05, 0.8, 0.2, 
            0.8, 0.75, 0.65, 0.7, 0.65, 
            0.5, 0.18, 0.3, 0.18, 0.02, 
            0.05, 0.1, 0.1, 0.15, 0.25, 
            0.2, 0.3, 0.4, 0.8, 0.6, 
            0.8
        )
        
        pgm_function3 = pgm_factor3.set_dense()
        pgm_function3.set_flat(
            0.8, 0.65, 0.45, 0.1, 0.05, 
            0.05, 0.1, 0.3, 0.5
        )
        
        pgm_function4 = pgm_factor4.set_dense()
        pgm_function4.set_flat(
            0.9, 0.14, 0.15, 0.05, 0.05, 
            0.05, 0.05, 0.05, 0.24, 0.03, 
            0.8, 0.01, 0.02, 0.22, 0.02, 
            0.05, 0.15, 0.33, 0.03, 0.02, 
            0.79, 0.15, 0.08, 0.4, 0.05, 
            0.05, 0.03, 0.01, 0.02, 0.04, 
            0.7, 0.5, 0.4, 0.05, 0.05, 
            0.34, 0.03, 0.02, 0.01, 0.08, 
            0.15, 0.13, 0.8, 0.7, 0.06
        )
        
        pgm_function5 = pgm_factor5.set_dense()
        pgm_function5.set_flat(
            0.2, 0.05, 0.4, 0.2, 0.8, 
            0.6, 0.8, 0.95, 0.6, 0.8, 
            0.2, 0.4
        )
        
        pgm_function6 = pgm_factor6.set_dense()
        pgm_function6.set_flat(0.9, 0.05, 0.1, 0.95)
        
        pgm_function7 = pgm_factor7.set_dense()
        pgm_function7.set_flat(
            0.1, 0.3, 0.5, 0.4, 0.5, 
            0.6, 0.3, 0.6, 0.4, 0.5, 
            0.45, 0.35, 0.6, 0.1, 0.1, 
            0.1, 0.05, 0.05
        )
        
        pgm_function8 = pgm_factor8.set_dense()
        pgm_function8.set_flat(
            0.1, 0.3, 0.5, 0.3, 0.6, 
            0.4, 0.6, 0.1, 0.1
        )
        
        pgm_function9 = pgm_factor9.set_dense()
        pgm_function9.set_flat(
            0.9, 0.9, 0.1, 0.1, 0.1, 
            0.9
        )
        
        pgm_function10 = pgm_factor10.set_dense()
        pgm_function10.set_flat(
            0.8, 0.1, 0.1, 0.08, 0.08, 
            0.06, 0.8, 0.02, 0.02, 0.02, 
            0.06, 0.02, 0.8, 0.1, 0.1, 
            0.02, 0.02, 0.02, 0.6, 0.1, 
            0.06, 0.06, 0.06, 0.2, 0.7
        )
        
        pgm_function11 = pgm_factor11.set_dense()
        pgm_function11.set_flat(
            0.2, 0.03061224, 0.3, 0.33673469, 0.25, 
            0.29591837, 0.15, 0.23469388, 0.05, 0.05102041, 
            0.05, 0.05102041
        )
        
        pgm_function12 = pgm_factor12.set_dense()
        pgm_function12.set_flat(0.8, 0.1, 0.2, 0.9)
        
        pgm_function13 = pgm_factor13.set_dense()
        pgm_function13.set_flat(
            0.95, 0.85, 0.8, 0.7, 0.7, 
            0.25, 0.8, 0.8, 0.8, 0.7, 
            0.9, 0.8, 0.03, 0.1, 0.15, 
            0.2, 0.15, 0.25, 0.15, 0.15, 
            0.15, 0.2, 0.08, 0.15, 0.02, 
            0.05, 0.05, 0.1, 0.15, 0.5, 
            0.05, 0.05, 0.05, 0.1, 0.02, 
            0.05
        )
        
        pgm_function14 = pgm_factor14.set_dense()
        pgm_function14.set_flat(
            0.1, 0.1, 0.1, 0.9, 0.05, 
            0.1, 0.9, 0.9, 0.9, 0.1, 
            0.95, 0.9
        )
        
        pgm_function15 = pgm_factor15.set_dense()
        pgm_function15.set_flat(
            0.15, 0.1, 0.8, 1.0, 0.33, 
            0.2, 0.05, 0.8, 0.2, 0.0, 
            0.33, 0.4, 0.8, 0.1, 0.0, 
            0.0, 0.34, 0.4
        )
        
        pgm_function16 = pgm_factor16.set_dense()
        pgm_function16.set_flat(
            0.4, 0.02, 0.02, 0.01, 0.01, 
            0.4, 0.43, 0.09, 0.16, 0.02, 
            0.03, 0.53, 0.15, 0.09, 0.8, 
            0.95, 0.95, 0.05, 0.02, 0.8, 
            0.02, 0.02, 0.01, 0.02
        )
        
        pgm_function17 = pgm_factor17.set_dense()
        pgm_function17.set_flat(
            0.6, 0.8, 0.8, 0.8, 0.1, 
            0.03, 0.1, 0.05, 0.05, 0.05, 
            0.6, 0.25, 0.3, 0.15, 0.15, 
            0.15, 0.3, 0.72
        )
        
        pgm_function18 = pgm_factor18.set_dense()
        pgm_function18.set_flat(
            0.3, 0.2, 0.15, 0.1, 0.3, 
            0.7, 0.65, 0.05, 0.8, 0.85, 
            0.1, 0.1, 0.05, 0.75, 0.05, 
            0.05, 0.6, 0.2
        )
        
        pgm_function19 = pgm_factor19.set_dense()
        pgm_function19.set_flat(
            0.4, 0.3, 0.2, 0.3, 0.7, 
            0.7, 0.6, 0.7, 0.8, 0.7, 
            0.3, 0.3
        )
