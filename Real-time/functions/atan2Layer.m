classdef atan2Layer < nnet.layer.Layer


    methods
        function layer = atan2Layer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            
            % Set number of inputs.
            layer.NumInputs = 2;
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "atan2Layer";
            
        end
        
        function Z = predict(~, X1, X2)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            
            Z = atan2(X2,X1);
        end

    end
end